"""Layer-streaming PyTorch forward pass for qwen35-family GGUFs (hybrid Gated-DeltaNet + gated full attention).

Weights are dequantized one layer at a time, so a 27B model needs only ~1 GB of weights resident (fp16 on GPU),
plus activations. Several candidate edits can be evaluated in the SAME stream: each layer is dequantized once and
every candidate gets its own (edited) copy of the residual writers. Used for:
  - capturing residual-stream vectors at chosen positions (directions, diagnostics)
  - deterministic proxy scores: NLL of continuations, log-probs of chosen tokens, greedy agreement (validation)

Generalisation notes vs the 0.8B port (validated against ollama logits on Qwen3.5-0.8B):
  - linear-attention value heads may outnumber key heads (Qwen3.8-27B: 48 vs 16). Which value head pairs with which
    key head ("interleave": j -> j // ratio, or "tile": j -> j % G) is checked by `validate` against ollama.
  - untied output head (`output.weight`) is used when present.
"""
import numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer
from .gguf_io import GGUF, writers
from .edits import ops_for, apply_torch


class Qwen35Stream:
    def __init__(self, gguf_path, tokenizer_path, device="auto", vgroup="interleave", base_dir="", lm_cache_gb=3.0):
        self.g = GGUF(gguf_path); m = self.g.meta
        self.L = int(m("block_count"))
        rec = m("attention.recurrent_layers")
        fai = int(m("full_attention_interval", 4))
        self.rec = [bool(x) for x in rec] if rec is not None else [(l + 1) % fai != 0 for l in range(self.L)]
        self.H, self.KV, self.hd = int(m("attention.head_count")), int(m("attention.head_count_kv")), int(m("attention.key_length"))
        self.rd, self.theta = int(m("rope.dimension_count")), float(m("rope.freq_base"))
        self.eps = float(m("attention.layer_norm_rms_epsilon", 1e-6))
        self.G, self.dk = int(m("ssm.group_count")), int(m("ssm.state_size"))
        self.Hv, self.inner = int(m("ssm.time_step_rank")), int(m("ssm.inner_size"))
        self.dv = self.inner // self.Hv
        self.D = int(m("embedding_length"))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = torch.device(device)
        self.wdt = torch.float16 if self.dev.type == "cuda" else torch.float32
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.vgroup, self.base_dir, self.lm_cache_gb = vgroup, base_dir, lm_cache_gb
        self.inv = 1.0 / (self.theta ** (torch.arange(0, self.rd, 2, dtype=torch.float64, device=self.dev) / self.rd))
        self.lm_name = "output.weight" if "output.weight" in self.g.T else "token_embd.weight"
        self._lm = None

    # ------------------------------------------------------------------ basics
    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids

    def enc_offsets(self, s):
        e = self.tok.encode(s, add_special_tokens=False)
        return e.ids, e.offsets

    def rms(self, x, w):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * w

    @staticmethod
    def l2n(x):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    def lin(self, x, W):
        return (x.to(W.dtype) @ W.T).float()

    def _layer(self, l):
        p = f"blk.{l}."; W = {}
        for n in self.g.T:
            if n.startswith(p):
                a = self.g.deq(n)
                if not a.flags.writeable:
                    a = a.copy()
                mat = a.ndim == 2 and not n.endswith("ssm_conv1d.weight")
                W[n] = torch.as_tensor(a, device=self.dev, dtype=self.wdt if mat else torch.float32)
        return W

    # ------------------------------------------------------------------ blocks
    def deltanet(self, l, x, W):
        p = f"blk.{l}."; B, T, _ = x.shape; G, dk, Hv, dv = self.G, self.dk, self.Hv, self.dv
        qkv = self.lin(x, W[p + "attn_qkv.weight"]); z = self.lin(x, W[p + "attn_gate.weight"])
        b = self.lin(x, W[p + "ssm_beta.weight"]); a = self.lin(x, W[p + "ssm_alpha.weight"])
        cw = W[p + "ssm_conv1d.weight"]; K = cw.shape[1]
        pad = F.pad(qkv, (0, 0, K - 1, 0))
        conv = F.silu(sum(pad[:, j:j + T] * cw[:, j] for j in range(K)))
        q, k, v = conv.split([G * dk, G * dk, Hv * dv], -1)
        q = self.l2n(q.reshape(B, T, G, dk)) / dk ** 0.5; k = self.l2n(k.reshape(B, T, G, dk)); v = v.reshape(B, T, Hv, dv)
        if Hv != G:
            r = Hv // G
            if self.vgroup == "interleave":
                q, k = q.repeat_interleave(r, 2), k.repeat_interleave(r, 2)
            else:
                q, k = q.repeat(1, 1, r, 1), k.repeat(1, 1, r, 1)
        beta = torch.sigmoid(b)
        decay = torch.exp(W[p + "ssm_a"] * F.softplus(a + W[p + "ssm_dt.bias"]))
        S = x.new_zeros(B, Hv, dk, dv); out = torch.empty_like(v)
        for t in range(T):
            S = S * decay[:, t, :, None, None]
            kv = torch.einsum("bhk,bhkv->bhv", k[:, t], S)
            S = S + k[:, t, :, :, None] * ((v[:, t] - kv) * beta[:, t, :, None])[:, :, None, :]
            out[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], S)
        o = self.rms(out, W[p + "ssm_norm.weight"]) * F.silu(z.reshape(B, T, Hv, dv))
        return self.lin(o.reshape(B, T, -1), W[p + "ssm_out.weight"])

    def rope(self, x):
        T = x.shape[1]; d = self.rd; h = d // 2
        ang = torch.outer(torch.arange(T, device=self.dev, dtype=torch.float64), self.inv)
        cos = torch.cat([ang.cos(), ang.cos()], -1).float()[None, :, None]; sin = torch.cat([ang.sin(), ang.sin()], -1).float()[None, :, None]
        xr, xp = x[..., :d], x[..., d:]
        rot = torch.cat([-xr[..., h:], xr[..., :h]], -1)
        return torch.cat([xr * cos + rot * sin, xp], -1)

    def attention(self, l, x, W):
        p = f"blk.{l}."; B, T, _ = x.shape; H, KV, hd = self.H, self.KV, self.hd
        qg = self.lin(x, W[p + "attn_q.weight"]).reshape(B, T, H, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:]
        k = self.lin(x, W[p + "attn_k.weight"]).reshape(B, T, KV, hd); v = self.lin(x, W[p + "attn_v.weight"]).reshape(B, T, KV, hd)
        q = self.rope(self.rms(q, W[p + "attn_q_norm.weight"])); k = self.rope(self.rms(k, W[p + "attn_k_norm.weight"]))
        k = k.repeat_interleave(H // KV, 2); v = v.repeat_interleave(H // KV, 2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        o = o.transpose(1, 2) * torch.sigmoid(gate)
        return self.lin(o.reshape(B, T, -1), W[p + "attn_output.weight"])

    def block(self, l, x, W):
        p = f"blk.{l}."
        h = self.rms(x, W[p + "attn_norm.weight"])
        x = x + (self.deltanet(l, h, W) if self.rec[l] else self.attention(l, h, W))
        h = self.rms(x, W[p + "post_attention_norm.weight"])
        return x + self.lin(F.silu(self.lin(h, W[p + "ffn_gate.weight"])) * self.lin(h, W[p + "ffn_up.weight"]), W[p + "ffn_down.weight"])

    # ------------------------------------------------------------------ output head
    def _lm_chunks(self, chunk=32768):
        V = self.g.shape(self.lm_name)[0]
        if self._lm is None and V * self.D * (2 if self.wdt == torch.float16 else 4) < self.lm_cache_gb * 1e9:
            lm = torch.empty(V, self.D, device=self.dev, dtype=self.wdt)
            for s in range(0, V, chunk):  # fill in row chunks: whole-matrix dequantization needs ~3x the memory
                e = min(V, s + chunk); lm[s:e] = torch.as_tensor(self.g.deq(self.lm_name, rows=np.arange(s, e)), device=self.dev, dtype=self.wdt)
            self._lm = lm
        for s in range(0, V, chunk):
            e = min(V, s + chunk)
            Wc = self._lm[s:e] if self._lm is not None else torch.as_tensor(self.g.deq(self.lm_name, rows=np.arange(s, e)), device=self.dev, dtype=self.wdt)
            yield s, e, Wc

    def _head(self, h, targets, picks, want_argmax):
        """h (N, D) final-normed hidden. targets: (N,) token ids or -1; picks: list of id lists per row or None.
        returns logsumexp-normalised target log-probs, pick log-probs, argmax ids"""
        N = h.shape[0]; lse = torch.full((N,), -float("inf"), device=self.dev)
        tgt = torch.full((N,), float("nan"), device=self.dev); amax_v = torch.full((N,), -float("inf"), device=self.dev)
        amax_i = torch.zeros(N, dtype=torch.long, device=self.dev)
        pk = [torch.full((len(p),), float("nan"), device=self.dev) if p else None for p in picks]
        T = torch.as_tensor(targets, device=self.dev)
        for s, e, Wc in self._lm_chunks():
            z = (h.to(Wc.dtype) @ Wc.T).float()
            lse = torch.logaddexp(lse, torch.logsumexp(z, -1))
            inr = (T >= s) & (T < e)
            if inr.any():
                rows = inr.nonzero().squeeze(-1); tgt[rows] = z[rows, T[rows] - s]
            if want_argmax:
                v, i = z.max(-1); better = v > amax_v; amax_v[better] = v[better]; amax_i[better] = i[better] + s
            for j, p in enumerate(picks):
                if p:
                    for q, tid in enumerate(p):
                        if s <= tid < e:
                            pk[j][q] = z[j, tid - s]
        tgt_lp = (tgt - lse).cpu().numpy()
        pk_lp = [None if p is None else (p - lse[j]).cpu().numpy() for j, p in enumerate(pk)]
        return tgt_lp, pk_lp, amax_i.cpu().numpy()

    # ------------------------------------------------------------------ main entry
    @torch.no_grad()
    def run(self, seqs, cands=((),), capture=None, score=None, bs_tokens=None, offload=False):
        """seqs: list of token-id lists. cands: list of spec lists (each evaluated in the same layer stream).
        capture: per seq dict {"pos": [int,...]} and/or {"span": (s, e)} -> residual after every layer
        score:   per seq dict {"nll": (s, e)} (NLL of tokens s..e-1), {"pick": (pos, [ids])} (log-probs at pos),
                 {"argmax": (s, e)} (greedy predictions for tokens s..e-1)
        returns list (per candidate) of dicts with keys pos {l: (n_pos, D)}, span {l: (n_seq, D)}, nll [(sum, n)],
        pick [array], argmax [array]"""
        cands = [list(c) for c in cands]; n = len(seqs)
        if bs_tokens is None:  # CPU float32 linear-attention intermediates are ~20 KB per token per batch
            bs_tokens = 4096 if self.dev.type == "cpu" else 16384
        order = sorted(range(n), key=lambda i: len(seqs[i]))
        batches, cur, cur_tok = [], [], 0
        for i in order:
            Ti = len(seqs[i])
            if cur and (len(cur) + 1) * max(Ti, cur_tok) > bs_tokens:
                batches.append(cur); cur, cur_tok = [], 0
            cur.append(i); cur_tok = max(cur_tok, Ti)
        if cur:
            batches.append(cur)
        emb = self._embed(seqs)
        xs = {}
        for c in range(len(cands)):
            for b, idx in enumerate(batches):
                Tm = max(len(seqs[i]) for i in idx)
                x = torch.zeros(len(idx), Tm, self.D, device=self.dev)
                for j, i in enumerate(idx):
                    x[j, :len(seqs[i])] = emb[i]
                xs[c, b] = x.cpu() if offload else x
        del emb
        res = [dict(pos={}, span={}, nll=[None] * n, pick=[None] * n, argmax=[None] * n) for _ in cands]
        cap = capture or [None] * n
        for l in range(self.L):
            W = self._layer(l)
            per_c = []
            for specs in cands:
                Wc = dict(W)
                for name in writers(self.g, l):
                    ops = ops_for(name, specs, self.base_dir)
                    if ops:
                        Wc[name] = apply_torch(W[name], ops)
                per_c.append(Wc)
            for c in range(len(cands)):
                posv, spanv = [None] * n, [None] * n
                for b, idx in enumerate(batches):
                    x = xs[c, b].to(self.dev)
                    x = self.block(l, x, per_c[c])
                    for j, i in enumerate(idx):
                        if cap[i]:
                            if cap[i].get("pos"):
                                posv[i] = x[j, cap[i]["pos"]].cpu().numpy()
                            if cap[i].get("span"):
                                s, e = cap[i]["span"]; spanv[i] = x[j, s:e].mean(0).cpu().numpy()
                    xs[c, b] = x.cpu() if offload else x
                if any(v is not None for v in posv):
                    res[c]["pos"][l] = np.concatenate([v for v in posv if v is not None])
                if any(v is not None for v in spanv):
                    res[c]["span"][l] = np.stack([v for v in spanv if v is not None])
            del W, per_c
        if score:
            onorm = torch.as_tensor(self.g.deq("output_norm.weight"), device=self.dev)
            for c, specs in enumerate(cands):
                on = onorm
                for op in ops_for("output_norm.weight", specs, self.base_dir):
                    on = on * op[1]
                for b, idx in enumerate(batches):
                    x = xs[c, b].to(self.dev)
                    rows, tg, pk, meta = [], [], [], []
                    for j, i in enumerate(idx):
                        sc = score[i]
                        if not sc:
                            continue
                        if "nll" in sc or "argmax" in sc:
                            s, e = sc.get("nll") or sc.get("argmax")
                            for t in range(s, e):
                                rows.append((j, t - 1)); tg.append(seqs[i][t]); pk.append(None); meta.append((i, "seq"))
                        if "pick" in sc:
                            pos, ids = sc["pick"]
                            rows.append((j, pos)); tg.append(-1); pk.append(list(ids)); meta.append((i, "pick"))
                        for pos, ids in sc.get("picks", []):   # several positions of one sequence in one pass
                            rows.append((j, pos)); tg.append(-1); pk.append(list(ids)); meta.append((i, "picks"))
                    if not rows:
                        continue
                    h = self.rms(x[[r[0] for r in rows], [r[1] for r in rows]], on)
                    step = 1024 if self.dev.type == "cpu" else 4096   # rows x 32k-vocab logits per slice: ~128 MB on CPU
                    for s0 in range(0, len(rows), step):
                        sl = slice(s0, s0 + step)
                        lp, plp, am = self._head(h[sl], tg[sl], pk[sl], any("argmax" in (score[m[0]] or {}) for m in meta[sl]))
                        for q, (i, kind) in enumerate(meta[sl]):
                            if kind == "pick":
                                res[c]["pick"][i] = plp[q]
                            elif kind == "picks":
                                res[c].setdefault("picks", [None] * n)
                                res[c]["picks"][i] = (res[c]["picks"][i] or []) + [plp[q]]
                            else:
                                if "nll" in score[i]:
                                    a0 = res[c]["nll"][i] or (0.0, 0); res[c]["nll"][i] = (a0[0] - float(lp[q]), a0[1] + 1)
                                if "argmax" in score[i]:
                                    res[c]["argmax"][i] = np.append(res[c]["argmax"][i] if res[c]["argmax"][i] is not None else np.array([], int), am[q])
        return res

    def _embed(self, seqs):
        ids = sorted({t for s in seqs for t in s}); idx = {t: i for i, t in enumerate(ids)}
        E = torch.as_tensor(self.g.deq("token_embd.weight", rows=np.array(ids)), device=self.dev)
        return [E[torch.tensor([idx[t] for t in s], device=self.dev)] for s in seqs]

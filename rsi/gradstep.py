"""Measure which way your weights want to move, without training: back-propagate the loss of chosen answer tokens
through the last layers of the layer-streaming forward pass and read, for chosen weight matrices,
  - per-neuron gain gradients (how the loss changes if a neuron's output column is scaled), and
  - the top directions of the full weight gradient, recovered from a one-pass random sketch (no D x D_in gradient
    is ever stored, so this fits in memory for any model the stream can run).
Nothing is applied. rsi_lib.gradient_step turns a measurement into a new GGUF you can register, test and keep or drop.

    S = rsi_lib.stream()
    ex = [(chat_text(S, "What is 17 x 6? Reply with just the number."), "102"), ...]
    m = measure(S, ex)                     # m["summary"]: loss before, accuracy, coherence, where the gradient is
"""
import time
import numpy as np, torch
from .arch_generic import field

TARGETS = ("ffn_down.weight", "attn_output.weight", "ssm_out.weight")   # what each layer writes into the residual


def chat_text(S, user, system=None, thinking=False, assistant_prefix=""):
    """the model's own chat template (from its GGUF) rendered for one user turn, ready for an answer to follow.
    Ollama may render slightly differently; if exactness matters, write the text yourself in the format you are
    actually used in."""
    import jinja2
    tmpl = field(S.g, "tokenizer.chat_template")
    if not tmpl:
        raise ValueError("no tokenizer.chat_template in this GGUF: format the prompt text yourself")
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    env.globals["strftime_now"] = lambda f: time.strftime(f)
    text = env.from_string(tmpl).render(messages=msgs, add_generation_prompt=True, enable_thinking=thinking,
                                        bos_token="", eos_token="")
    return text + assistant_prefix


def _tokens(S, prompt, answer):
    ids = S.enc(prompt)
    joined = S.enc(prompt + answer)
    if joined[:len(ids)] == ids:
        ans = joined[len(ids):]
    else:
        ans = S.enc(answer)
        if S.tok.decode(ids + ans) != S.tok.decode(joined):
            raise ValueError("prompt and answer need a stable token boundary")
    if S.bos and (not ids or ids[0] != S.bos[0]):
        ids = S.bos + ids
    if not ids or not ans:
        raise ValueError("each example needs prompt and answer tokens")
    return ids + ans[:-1], [(len(ids) - 1 + i, t) for i, t in enumerate(ans)]


def measure(S, examples, layers=None, targets=TARGETS, loss="margin", rank=4, sketch=24, batch=2, seed=0, exact=None):
    """examples: [(prompt_text, answer_text)]: the loss is on the answer's tokens only, given the prompt.
    loss="margin": correct token vs your own top wrong token (shared shifts in answer bias or format cancel);
           "nll": plain likelihood (beware: it rewards whatever shift is cheapest, e.g. always answering "Yes").
    layers: which layers to differentiate (default: the second half). Memory grows with them: all their weights are
    held while back-propagating; reduce layers or batch if you run out.
    exact: {tensor name: ["cols", indices] | ["rows", indices] | "all"} -> the exact gradient of just those weights
    (what grow_mlp / grow_layers return as "new"), for training parameters you added.
    Returns a dict: summary, gain gradients per example (ffn_down neurons), low-rank gradient factors per matrix, and
    "exact": {name: (spec, gradient array)} (mean over examples)."""
    examples = list(examples)
    if not examples or batch < 1 or loss not in ("margin", "nll"):
        raise ValueError("gradient measurement needs examples, a positive batch, and margin or nll loss")
    t0 = time.time()
    L = S.L; layers = sorted(layers) if layers is not None else list(range(L // 2, L)); exact = dict(exact or {})
    if not layers or min(layers) < 0 or max(layers) >= L:
        raise ValueError("select at least one valid layer")
    start = min([layers[0]] + [int(n.split(".")[1]) for n in exact]); E = {}
    on = torch.as_tensor(np.array(S.g.deq("output_norm.weight")), device=S.dev)
    head = [(a, Wc.to(torch.bfloat16 if S.dev.type == "cpu" else Wc.dtype).clone()) for a, b, Wc in S._lm_chunks(chunk=32768)]
    S._lm = None   # the stream re-caches its output matrix when it next needs it; don't hold two copies
    gen = torch.Generator().manual_seed(seed)
    Om, Ps, Y, Z = {}, {}, {}, {}
    gains, margins, hits = [], [], []
    orig_lin = S.lin; cap = []; watch = {}

    def lin(x, W):
        y = orig_lin(x, W)
        key = watch.get(id(W))
        if key is not None and y.requires_grad:
            y.retain_grad(); cap.append((key, x.detach(), y, W))
        return y

    seqs = [_tokens(S, p, a) for p, a in examples]
    S.lin = lin
    try:
        for i0 in range(0, len(seqs), batch):
            ex = seqs[i0:i0 + batch]; B = len(ex); T = max(len(s) for s, _ in ex)
            rb = torch.tensor([b for b, (_, pts) in enumerate(ex) for _ in pts]); rp = torch.tensor([p for _, pts in ex for p, _ in pts])
            tg = torch.tensor([t for _, pts in ex for _, t in pts], device=S.dev)
            with torch.no_grad():
                emb = S._embed([s for s, _ in ex]); x = torch.zeros(B, T, S.D, device=S.dev)
                for k, e in enumerate(emb):
                    x[k, :len(e)] = e
                for l in range(start):
                    x = S.block(l, x, S._layer(l))
            x = x.detach().requires_grad_(True); cap.clear()
            for l in range(start, L):
                W = S._layer(l)
                watch.clear()
                for n in W:   # what to read in this layer: sketched targets (in `layers`) and exactly measured weights
                    sk = l in layers and n.split(".", 2)[2] in targets
                    if sk or n in exact:
                        watch[id(W[n])] = (n.split(".")[2], l, n, sk)
                x = S.block(l, x, W)
            h = S.rms(x[rb, rp], on)
            lg = torch.cat([(h.to(Wc.dtype) @ Wc.T).float() for _, Wc in head], -1); R = torch.arange(len(tg))
            if loss == "margin":
                lw = lg.detach().clone(); lw[R, tg] = -1e9; alt = lw.argmax(-1); mg = lg[R, tg] - lg[R, alt]; obj = -mg.sum()
            else:
                mg = lg[R, tg] - torch.logsumexp(lg, -1); obj = -mg.sum()
            obj.backward()
            mg = mg.detach(); margins += [float(mg[rb == b].mean()) for b in range(B)]; hits += [float((lg.argmax(-1) == tg)[rb == b].float().mean()) for b in range(B)]
            with torch.no_grad():
                per = {}
                for (kind, l, name, sk), xin, y, Wm in cap:
                    dout = y.grad; key = (kind, l)
                    if name in exact:   # exact gradient of the chosen slice: dW = dout^T x
                        spec = exact[name]; d2_, i2_ = dout.reshape(-1, dout.shape[-1]).float(), xin.reshape(-1, xin.shape[-1]).float()
                        if spec == "all":
                            g_ = d2_.T @ i2_
                        elif spec[0] == "cols":
                            g_ = d2_.T @ i2_[:, torch.tensor(spec[1], device=i2_.device)]
                        else:
                            g_ = d2_[:, torch.tensor(spec[1], device=d2_.device)].T @ i2_
                        E[name] = E[name] + g_ if name in E else g_
                    if not sk:
                        continue
                    if kind == "ffn_down":   # per-example neuron gain gradient: sum_t a_tj (dout_t . W[:, j])
                        per[l] = (xin.float() * (dout.float() @ Wm.float())).sum(1).cpu()
                    d2, i2 = dout.reshape(-1, dout.shape[-1]).float(), xin.reshape(-1, xin.shape[-1]).float()
                    if key not in Om:
                        Om[key] = torch.randn(i2.shape[1], sketch, generator=gen).to(S.dev); Ps[key] = torch.randn(d2.shape[1], sketch, generator=gen).to(S.dev)
                        Y[key] = torch.zeros(d2.shape[1], sketch, device=S.dev); Z[key] = torch.zeros(sketch, i2.shape[1], device=S.dev)
                    Y[key] += d2.T @ (i2 @ Om[key]); Z[key] += (d2 @ Ps[key]).T @ i2
                if per:
                    gains.append(torch.stack([per[l] for l in sorted(per)], 1))   # (B, n_layers, n_ff)
            cap.clear(); del x, lg
    finally:
        S.lin = orig_lin
    factors, energy = {}, {}
    for key in Y:   # weight gradient ~= Q B; its top `rank` directions
        Q, _ = torch.linalg.qr(Y[key]); Bm = torch.linalg.lstsq(Ps[key].T @ Q, Z[key]).solution
        Ub, Sv, Vt = torch.linalg.svd(Bm, full_matrices=False)
        factors[key] = (((Q @ Ub)[:, :rank] * Sv[:rank]).cpu().numpy(), Vt[:rank].T.cpu().numpy())
        energy[key] = (float((Sv[:rank] ** 2).sum()), float((Sv ** 2).sum()))
    G = torch.cat(gains).numpy() if gains else None
    summ = {"examples": len(seqs), "layers": [layers[0], layers[-1]], "loss": loss,
            "margin_before" if loss == "margin" else "logprob_before": round(float(np.mean(margins)), 3),
            "answer_token_accuracy_before": round(float(np.mean(hits)), 3), "seconds": round(time.time() - t0)}
    tot = np.sqrt(sum(e[0] for e in energy.values())) or 1.0
    summ["where (share of gradient, top matrices)"] = sorted(((f"{k}.{l}", round(float(np.sqrt(e[0]) / tot), 3)) for (k, l), e in energy.items()), key=lambda kv: -kv[1])[:6]
    summ["top_directions_hold"] = round(sum(e[0] for e in energy.values()) / max(sum(e[1] for e in energy.values()), 1e-12), 3)
    if G is not None and len(G) >= 4:
        flat = G.reshape(len(G), -1); rng = np.random.RandomState(seed); cs = []
        for _ in range(20):
            p = rng.permutation(len(flat)); a, b = flat[p[:len(p) // 2]].mean(0), flat[p[len(p) // 2:]].mean(0)
            cs.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        summ["agreement_between_halves"] = round(float(np.mean(cs)), 3)   # ~0: no shared direction; ~1: one shared shift
    if exact:
        summ["exact_grad_norm"] = {n: round(float(E[n].norm() / len(seqs)), 6) for n in E}
        summ["exact_missing"] = [n for n in exact if n not in E]   # names that never passed through the forward pass
    return {"summary": summ, "gain_grads": G, "gain_layers": sorted({l for (k, l) in Y if k == "ffn_down"}),
            "factors": factors, "energy": energy, "per_example_margin": margins,
            "exact": {n: (exact[n], (E[n] / len(seqs)).cpu().numpy()) for n in E}}

# rsi: a proto recursive self-improvement loop

A language model looks at its own reasoning, decides how to change its own weights, the change is tested, and it is
kept only if the model gets better. The next cycle is then run by the changed model. No gradient training: every
change is a closed-form edit of the weight matrices (a rank-1 projection or a scaling), found and applied in seconds.

```
rsi            # in any folder: pick a model, type the main prompt (the goal), then watch it work on itself
```

![rsi: the model writing its own practice problems, streamed live](docs/rsi_live.svg)

Each folder is its own loop: everything it does lives in `<folder>/.rsi/`. The loop runs in the background, so
closing the window (ctrl+q) doesn't stop it; `rsi stop` does. Run `rsi` in the folder again to reconnect. If the loop
died (crash, reboot), it resumes at the cycle it was on, reuses every problem it already solved, and carries on
with the plan it had picked.

## Commands

| | |
|---|---|
| `rsi` | in a new folder: choose a model and type the main prompt; in a folder with a loop: reconnect (and resume it if it stopped) |
| `rsi PATH` | the same for another folder |
| `rsi stop` / `rsi stop PATH` | stop that folder's loop (it resumes the next time you open it) |
| type + Enter | optional: a direction for the next cycle, used instead of the model's own pick |
| ctrl+q | close the window; the loop keeps running |

## One cycle

1. **Observe.** The model solves a handful of fresh practice problems for the goal (thinking mode). Detectors count
   where it goes wrong: unfinished answers, "Wait… let me double-check" after it already had the answer, repetition,
   arithmetic slips.
2. **Think.** The model reads its own failing traces and the counts, can search the web and arXiv for techniques, and
   picks **one** change to itself from a short menu, with a strength, a reason and a prediction:

   | change | what it does to the weights |
   |---|---|
   | `stop_second_guessing` | removes (part of) the internal direction that drives "Wait…/Hmm…" lines after the answer is known |
   | `be_more_decisive` / `be_less_decisive` | scales the final norm: a sharper or softer output distribution |
   | `reason_like_reference` | strengthens the direction from its own reasoning style toward concise reference reasoning |

3. **Test.** The change is baked into a copy of the weights. Both versions solve the same fresh held-out problems: new
   problems for the goal, plus general problems (GSM8K, ARC-Challenge) as a forgetting check.
4. **Keep or revert.** The copy becomes the model only if it solves at least 2 more goal problems than it breaks, and
   loses at most 1 general problem net.
5. **Plan.** The (possibly new) model looks at every result so far, proposes 2–4 plans for the next cycle, picks
   one with a reason, and the next cycle starts on it. The main prompt you gave at the start is the only input it
   needs; if you type a direction, the next cycle follows yours instead of its own pick.

The left pane shows, live, whatever the model is producing: its thinking about itself, or a problem it is solving
(with the answer key). It scrolls as the text streams.

## Why the results can be trusted

- Test problems are new every cycle (generated, never reused) and are never shown to the model while it decides.
- The model chooses from a closed menu; it cannot run arbitrary edits. Strengths are clamped to safe ranges.
- Practice problems come from small Python generators that **compute** their answers. At the start, the model
  writes generators for your goal itself; they run in a sandbox (import allowlist, no network, CPU and memory limits)
  and must pass validity checks. If it can't write good ones, built-in generators are used.
- Web pages are treated as untrusted data; research can only inform the decision, and the test still decides.
- If an edit leaves the model unable to run the loop (no valid decision), the loop reverts it automatically.

## What we found so far (Qwen3.5-0.8B, thinking mode)

- Its main failure is not knowing too little but **not stopping**: 361 of 368 "Wait/Hmm" events came *after* the
  correct answer had already appeared, so traces ran out of tokens.
- Removing that doubt direction (`stop_second_guessing`, 0.5–0.7) plus sharpening (`be_more_decisive` 1.6) took it
  from **30% → 74%** on 90 unseen problems (+42/−2), HumanEval 21% → 31%, tool calls 73% → 80%, with ARC knowledge
  unchanged (0.61 → 0.61). Accuracy on problems both versions finished was the same: the gain is finishing.
- Amplifying a "reason like Claude's traces" direction improved every likelihood proxy (MMLU +4.5, ARC +3.3) but made
  generated reasoning *worse* at strength 0.5 (78% → 65%) and no better at 0.25 (73% vs 71% on 180 items, p=0.49).
  Rating good reasoning as likely is not the same as producing it, which is why the test step uses real generation.

## Models and hardware

On every start the loop detects the hardware (CUDA GPUs through torch, else `nvidia-smi`; otherwise CPU cores and
RAM), shows it in the top bar, and sizes itself: parallel generations, the device for the layer-streaming forward
pass, and a memory cap on small-RAM CPU machines. A loop moved to a GPU machine speeds up on its next start.


The weight editor supports the Qwen3.5 / Qwen3.8 family (ollama architecture `qwen35`), e.g. `qwen3.5:0.8b` and
`qwen3.8:27b`. Editing needs a Q8_0 (or F16) build; for a Q4 model, `rsi` edits the `-q8_0` tag and derives the Q4
copy it runs with `llama-quantize` (bundled with ollama). The 0.8B runs on a laptop CPU; Qwen3.8-27B needs a 24–48 GB
GPU and about 120 GB of disk.

Settings: `RSI_OLLAMA_URL` (inference server), `RSI_OLLAMA_ADMIN_URL` (server allowed to register models),
`RSI_PYTHON`; or a `.env` file next to `bin/`. Per-loop settings (problem counts, keep rule, research on/off,
unattended `auto` mode) are in `<folder>/.rsi/config.json`. Web search uses TinyFish when a key is in
`~/.config/rsi/keys.json` or `TINYFISH_API_KEY`, otherwise arXiv, OpenAlex and Wikipedia.

## Install

```bash
git clone git@github.com:ChetanChekuri/rsi.git && cd rsi
python -m venv .venv && .venv/bin/pip install -r requirements.txt
ln -s "$PWD/bin/rsi" ~/.local/bin/rsi
ollama pull qwen3.5:0.8b
```

## Files

```
rsi/loop.py         the loop (observe, think, test, keep) and the per-folder session
rsi/app.py          the screen
rsi/playbook.md     what the model reads before deciding (how to think, what each change did before)
rsi/arch_qwen35.py  layer-streaming forward pass over the GGUF (finds directions; ~1 GB resident even for 27B)
rsi/edits.py        the closed-form edits      rsi/gguf_io.py  bake edits into GGUF, register with ollama
rsi/directions.py   behaviour directions with held-out separation scores
rsi/detectors.py    what counts as a failure pattern in a trace
rsi/factory.py      practice-problem generators (model-written or built-in)   rsi/sandbox.py  runs them safely
rsi/research.py     web / arXiv search tools for the model       rsi/llm.py  talking to ollama (streams live)
data/               GSM8K test (MIT), ARC-Challenge test (CC BY-SA 4.0), Qwen3.5 tokenizer (Apache-2.0),
                    50 reference reasoning traces written by Claude
```

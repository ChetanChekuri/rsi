# rsi: a proto recursive self-improvement loop

A language model works on its own weights. It gets a sandbox, its weights, and freedom to run any code it likes on
them. It can build new versions of itself, register them with ollama, try them out, and submit one as its successor.
A held-out test the model can't touch decides whether the successor replaces it. The next cycle is then run by
whichever model won.

```
rsi            # in any folder: pick a model, type the main prompt (the goal), then watch it work on itself
```

![rsi: the model's work streamed live](docs/rsi_live.svg)

Each folder is its own loop. Its state lives in `<folder>/.rsi/` and the model's files live in `<folder>/workspace/`.
The loop runs in the background, so closing the window (ctrl+q) doesn't stop it. `rsi stop` stops it. Run `rsi` in
the folder again to reconnect (or restart it). If the loop died (crash, reboot), it resumes at the cycle it was on.

## Commands

| | |
|---|---|
| `rsi` | in a new folder: choose a model and type the main prompt; in a folder with a loop: reconnect (and resume it unless it was stopped) |
| `rsi PATH` | the same for another folder |
| `rsi stop` / `rsi stop PATH` | stop that folder's loop; it stays stopped until you open `rsi` there again |
| type + Enter | talk to the model while it works (it reads your message at its next step); the message also becomes the next cycle's direction |
| ctrl+q | close the window; the loop keeps running |

## One cycle

1. **Practice.** The loop has the current model solve a few fresh practice problems for the goal (thinking mode) and
   saves the traces into the workspace. Detectors count where it went wrong: unfinished answers, "Wait… let me
   double-check" after it already had the answer, repetition, arithmetic slips.
2. **Work.** The model gets a brief (the goal, its practice results, failing excerpts, earlier cycles, its plan) and
   then works on its own with tools, up to 40 tool calls or 2 hours by default:

   | tool | what it does |
   |---|---|
   | `run_python`, `run_shell` | run any code in the workspace sandbox; numpy, torch and `rsi_lib` are available |
   | `read_file`, `write_file`, `list_files` | files in the workspace (notes, scripts, results, GGUFs) |
   | `web_search`, `arxiv_search`, `fetch_page` | research (TinyFish, arXiv); pages are treated as untrusted text |
   | `ollama_create` | register a GGUF it built as a runnable model (`rsi-<loop>:<name>`) |
   | `ollama_list`, `ollama_delete` | list its models; delete ones it made (never its current self or anything it didn't create) |
   | `practice` | have any of its models solve n fresh practice problems; the traces are saved for analysis |
   | `submit` | propose one of its models as its successor (once per cycle) |
   | `finish` | end the cycle with a summary, 2–4 plans for the next cycle, and its pick |

   `rsi_lib` gives the sandbox code a head start: `read_tensor`, `write_model` (a copy of itself with some tensors
   replaced), `stream()` (a layer-streaming forward pass for activations and losses), `practice_traces()`,
   `doubt_direction()`, `ablate`/`amplify`. The model can ignore all of it and write its own methods.
3. **Test.** `submit` runs the held-out test outside the sandbox. The candidate and the current model solve the same
   fresh problems for the goal, plus general problems (GSM8K, ARC-Challenge) as a forgetting check. The candidate is
   kept only if it solves at least 2 more goal problems than it breaks and loses at most 1 general problem net. If it
   is kept, it becomes the current model at once and finishes the cycle itself.
4. **Plan.** `finish` records the plans and the pick; the next cycle starts on that plan. If you typed a direction,
   the next cycle follows yours instead.

The left pane shows, live, whatever the model is producing: its thinking, its tool calls and their results, or a
problem it is solving (with the answer key). It scrolls as the text streams.

## The sandbox

Code runs under bubblewrap, inside a systemd scope:

- The workspace is the only writable place, and the only one that persists.
- The system, the rsi code, the Python environment and ollama's model files are read-only. The model reads its own
  weights from there.
- The home directory is hidden, so API keys and the loop's own state aren't visible.
- The network is off. Research goes through the web tools, which the loop runs itself.
- Memory, CPU and time are capped: 4 GB, half the cores and 30 minutes per run on a laptop.

Set `"sandbox_network": true` in `.rsi/config.json` to allow network access from code. GPUs are passed through when
present.

## Why the results can be trusted

- The judge isn't the model. `submit` runs the test in the loop's own process. The model only picks which of its
  models to submit, and it gets one submission per cycle.
- Test problems are new every cycle and are never shown to the model. Their seeds are offset by a secret stored in
  `.rsi/`, which the sandbox can't read, so the model can't regenerate its own test and train on it.
- Practice and test problems come from small Python generators that **compute** their answers. At the start, the
  model writes generators for your goal; they run in a separate strict sandbox (import allowlist, no files, no
  network) and must pass validity checks. If it can't write good ones, built-in generators are used.
- Web pages are untrusted data: they can inform what the model tries, and the test still decides.

## What we found so far (Qwen3.5-0.8B, thinking mode)

- Its main failure is not knowing too little but **not stopping**: 361 of 368 "Wait/Hmm" events came *after* the
  correct answer had already appeared, so traces ran out of tokens.
- Removing that doubt direction (rank-1 ablation, 0.5–0.7) plus sharpening the output (final norm × 1.6) took it
  from **30% → 74%** on 90 unseen problems (+42/−2), HumanEval 21% → 31%, tool calls 73% → 80%, with ARC knowledge
  unchanged (0.61 → 0.61). On 60 more fresh problems: 35% → 73% (+23/−0). Accuracy on problems both versions
  finished was the same: the gain is finishing.
- Amplifying a "reason like Claude's traces" direction improved every likelihood proxy (MMLU +4.5, ARC +3.3) but made
  generated reasoning *worse* at strength 0.5 (78% → 65%) and no better at 0.25 (73% vs 71% on 180 items, p=0.49).
  Rating good reasoning as likely is not the same as producing it, which is why the test uses real generation.

These findings are in the playbook the model reads at the start of every cycle.

## Models and hardware

On every start the loop detects the hardware (CUDA GPUs through torch, else `nvidia-smi`; otherwise CPU cores and
RAM), shows it in the top bar, and sizes itself: parallel generations, memory and CPU caps for the loop and the
sandbox, and the device for the layer-streaming forward pass. A loop moved to a GPU machine speeds up on its next
start.

The weight tools support the Qwen3.5 / Qwen3.8 family (ollama architecture `qwen35`), e.g. `qwen3.5:0.8b` and
`qwen3.8:27b`. In-place edits need a Q8_0 (or F16) build; for a Q4 model, `rsi` edits the `-q8_0` tag and derives the
Q4 copy it runs with `llama-quantize` (bundled with ollama). The 0.8B runs on a laptop CPU; Qwen3.8-27B needs a 24–48
GB GPU and about 120 GB of disk (each model it builds is another full copy, so it is told to clean up).

Settings: `RSI_OLLAMA_URL` (inference server), `RSI_OLLAMA_ADMIN_URL` (server allowed to register and delete models),
`RSI_PYTHON`; or a `.env` file next to `bin/`. Per-loop settings are in `<folder>/.rsi/config.json`: problem counts,
the keep rule (`min_gain`, `max_general_loss`), `max_steps`, `max_cycle_minutes`, `max_run_s`, `sandbox_mem`,
`sandbox_network`. Web search uses TinyFish when a key is in `~/.config/rsi/keys.json` or `TINYFISH_API_KEY`,
otherwise arXiv, OpenAlex and Wikipedia.

## Install

```bash
git clone git@github.com:ChetanChekuri/rsi.git && cd rsi
python -m venv .venv && .venv/bin/pip install -r requirements.txt
ln -s "$PWD/bin/rsi" ~/.local/bin/rsi
ollama pull qwen3.5:0.8b
```

The sandbox needs `bwrap` (`apt install bubblewrap`).

## Files

```
rsi/loop.py         the loop (practice, work, test, plan) and the per-folder session
rsi/tools.py        the model's tools          rsi/sandbox.py   the workspace sandbox (and the strict one for generators)
rsi/app.py          the screen                 rsi/playbook.md  what the model reads every cycle (tools, method, findings)
rsi_lib/            helpers importable inside the sandbox
rsi/arch_qwen35.py  layer-streaming forward pass over the GGUF (~1 GB resident even for 27B)
rsi/gguf_io.py      read/patch GGUF tensors, register with ollama    rsi/edits.py  closed-form edit operators
rsi/directions.py   behaviour directions with held-out separation scores
rsi/detectors.py    what counts as a failure pattern in a trace
rsi/factory.py      problem generators (model-written or built-in)
rsi/research.py     web / arXiv search        rsi/llm.py  talking to ollama (streams live)
data/               GSM8K test (MIT), ARC-Challenge test (CC BY-SA 4.0), Qwen3.5 tokenizer (Apache-2.0),
                    50 reference reasoning traces written by Claude
```

"""Runs a suite's grade.py inside the sandbox: python -m rsi.grade GRADE_PY IN.jsonl OUT.json"""
import importlib.util, json, sys


def main(gpath, inp, out):
    spec = importlib.util.spec_from_file_location("suite_grade", gpath); mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, __import__("os").path.dirname(gpath)); spec.loader.exec_module(mod)
    scores = []
    for l in open(inp):
        d = json.loads(l)
        try:
            s = min(1.0, max(0.0, float(mod.grade(d["problem"], d["response"]) or 0)))
        except Exception as e:  # noqa: BLE001
            s = 0.0; print(f"grade.py failed on {d['problem'].get('id')}: {type(e).__name__}: {e}", file=sys.stderr)
        scores.append(s)
    json.dump(scores, open(out, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:4])

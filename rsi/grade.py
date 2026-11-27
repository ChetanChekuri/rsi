"""Runs a suite's grade.py inside the sandbox: python -m rsi.grade GRADE_PY IN.jsonl OUT.json"""
import importlib.util, json, math, os, sys


def main(gpath, inp, out, suite_path=None):
    spec = importlib.util.spec_from_file_location("suite_grade", gpath); mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, suite_path or os.path.dirname(gpath)); spec.loader.exec_module(mod)
    scores = []
    with open(inp, encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            try:
                score = float(mod.grade(row["problem"], row["response"]))
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("score must be finite and in [0, 1]")
            except Exception as exc:
                raise ValueError(f"grade.py failed on {row['problem'].get('id')}") from exc
            scores.append(score)
    with open(out, "w", encoding="utf-8") as target:
        json.dump(scores, target)


if __name__ == "__main__":
    main(*sys.argv[1:5])

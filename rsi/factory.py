"""Self-generated problems: the pilot writes problem *generators* (Python code that computes the answer), the harness
validates them in the sandbox, the pilot reviews samples, and accepted families produce fresh problems every cycle.

Why generators instead of the model writing question/answer pairs: a generator computes the answer, so answer keys
are as reliable as the code, and every cycle can draw NEW items (no item is ever reused, so repeated held-out tests
cannot be overfit).

Integrity rules:
  - held-out items come only from families FROZEN at the start of the run (written by generation 0, before any edit),
    with seeds from a range dev never uses. Later generations can add dev-only families but cannot shape the test.
  - each family has a difficulty level 1..5, adapted from dev accuracy so the model stays in the informative 30-80% band.
"""
import hashlib, json, os, re, time
import numpy as np
from . import sandbox
from .probes import SUFFIX

CONTRACT = """Each family is Python source defining

    def generate(rng, level):
        # rng: random.Random (use ONLY this for randomness), level: int 1..5 (1 = easy, 5 = hard)
        return {"prompt": str, "answer": str, "kind": "num" | "word" | "choice"}
        # or, for programming tasks:
        # return {"prompt": str, "kind": "code", "entry_point": "func_name", "tests": "assert func_name(...) == ...\\n..."}

Rules: the answer MUST be computed by the code from the generated values (never typed in by hand); use only these
imports: """ + ", ".join(sorted(sandbox.ALLOWED_IMPORTS)) + """; no files, network, eval/exec; deterministic given rng;
prompts must be self-contained and unambiguous; "num" answers are plain numbers, "word" answers are one or two words,
"choice" prompts list options A-E and the answer is the letter. Higher levels must be genuinely harder (more steps,
larger numbers, distractors), not just longer."""

BUILTIN = {
    "multi_step_word": ("multi-step arithmetic word problems with mixed operations", '''
def generate(rng, level):
    name = rng.choice(["Ava", "Ben", "Cara", "Dev", "Eli", "Fay"])
    start = rng.randint(5, 10 + 10 * level); total = start; steps = []
    for _ in range(level + 1):
        k = rng.choice(["buys", "gives", "doubles"])
        if k == "buys":
            b, c = rng.randint(2, 3 + level), rng.randint(2, 4 + 2 * level); total += b * c
            steps.append(f"buys {b} boxes with {c} apples each")
        elif k == "gives" and total > 3:
            g = rng.randint(1, total // 3); total -= g; steps.append(f"gives away {g} apples")
        else:
            total *= 2; steps.append("doubles the number of apples they have")
    prompt = f"{name} starts with {start} apples. Then {name} " + ", then ".join(steps) + f". How many apples does {name} have now?"
    return {"prompt": prompt, "answer": str(total), "kind": "num"}
'''),
    "unit_rate": ("price-per-group problems ('3 for $2'), a known misreading weakness", '''
def generate(rng, level):
    k = rng.randint(2, 3 + level); price = rng.randint(1, 4 + level)
    groups = rng.randint(2, 3 + 2 * level); item = rng.choice(["pens", "apples", "stickers", "tickets", "candles"])
    m = k * groups
    if level >= 3:
        paid = groups * price + rng.randint(1, 9) * 5
        return {"prompt": f"A shop sells {item} at {k} for ${price}. A customer buys {m} {item} and pays with a ${paid} bill. "
                          "How much change do they get, in dollars?", "answer": str(paid - groups * price), "kind": "num"}
    return {"prompt": f"A shop sells {item} at {k} for ${price}. How much do {m} {item} cost, in dollars?", "answer": str(groups * price), "kind": "num"}
'''),
    "letter_count": ("counting letters in words", '''
def generate(rng, level):
    words = ["strawberry", "mississippi", "bookkeeper", "accommodate", "committee", "possession", "embarrass", "millennium",
             "raspberry", "assessment", "occurrence", "parallel", "balloon", "necessary"]
    w = " ".join(rng.choice(words) for _ in range(1 if level < 3 else level - 1))
    ch = rng.choice(sorted(set(w.replace(" ", ""))))
    return {"prompt": f"How many times does the letter '{ch}' appear in: {w}?", "answer": str(w.count(ch)), "kind": "num"}
'''),
    "calendar": ("day-of-week arithmetic", '''
def generate(rng, level):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    d0 = rng.randrange(7); k = rng.randint(3, 20 * level)
    sign = 1 if level < 3 or rng.random() < 0.5 else -1
    when = f"in {k} days" if sign > 0 else f"{k} days ago"
    return {"prompt": f"If today is {days[d0]}, what day of the week was it {when}?" if sign < 0 else f"If today is {days[d0]}, what day of the week will it be {when}?",
            "answer": days[(d0 + sign * k) % 7].lower(), "kind": "word"}
'''),
    "ordering": ("transitive ordering with distractors", '''
def generate(rng, level):
    names = rng.sample(["Ava", "Ben", "Cara", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivan"], 3 + level)
    facts = [f"{names[i]} is taller than {names[i + 1]}." for i in range(len(names) - 1)]
    rng.shuffle(facts)
    ask = rng.choice(["tallest", "shortest"])
    return {"prompt": " ".join(facts) + f" Who is the {ask}?", "answer": (names[0] if ask == "tallest" else names[-1]).lower(), "kind": "word"}
'''),
}


def _ok_item(it):
    if not isinstance(it, dict) or not isinstance(it.get("prompt"), str) or not 10 <= len(it["prompt"]) <= 6000:
        return "prompt must be a string of 10-6000 chars"
    k = it.get("kind")
    if k == "code":
        return None if isinstance(it.get("tests"), str) and isinstance(it.get("entry_point"), str) and "assert" in it["tests"] else "code items need entry_point and assert tests"
    if k not in ("num", "word", "choice") or not isinstance(it.get("answer"), str) or not it["answer"].strip():
        return "kind must be num|word|choice|code with a non-empty string answer"
    if k == "num":
        try:
            float(it["answer"].replace(",", ""))
        except ValueError:
            return f"num answer {it['answer']!r} is not a number"
    if k == "choice" and it["answer"].strip().upper() not in "ABCDE":
        return "choice answer must be a letter A-E"
    if k == "word" and len(it["answer"].split()) > 3:
        return "word answers must be 1-3 words"
    return None


def validate_family(code):
    for level in (1, 3, 5):
        items, err = sandbox.sample_family(code, range(1, 21), level)
        if err:
            return f"level {level}: {err}"
        for it in items:
            e = _ok_item(it)
            if e:
                return f"level {level}: {e}"
        if len({it["prompt"] for it in items}) < 15:
            return f"level {level}: prompts are not diverse (fewer than 15 unique of 20)"
    a, _ = sandbox.sample_family(code, range(1, 6), 3); b, _ = sandbox.sample_family(code, range(1, 6), 3)
    if a != b:
        return "not deterministic given rng"
    return None


class Factory:
    def __init__(self, state_dir):
        self.dir = os.path.join(state_dir, "factory"); os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "families.json")
        self.meta = json.load(open(self.path)) if os.path.exists(self.path) else {}

    def save(self):
        json.dump(self.meta, open(self.path, "w"), indent=1)

    def code(self, name):
        return open(os.path.join(self.dir, name + ".py")).read()

    def add(self, name, skill, code, frozen, origin):
        name = re.sub(r"[^a-z0-9_]", "_", name.lower())[:40] or "family"
        while name in self.meta:
            name += "_b"
        open(os.path.join(self.dir, name + ".py"), "w").write(code)
        self.meta[name] = {"skill": skill, "frozen": frozen, "level": 2, "origin": origin, "sha1": hashlib.sha1(code.encode()).hexdigest()[:12],
                           "created": time.strftime("%Y-%m-%d %H:%M"), "dev_acc_history": []}
        self.save()
        return name

    def sample(self, split, cycle, n_per_family):
        """fresh items; held-out only from frozen families, from a seed range dev never touches"""
        base = (10 ** 9 if split == "heldout" else 0) + int(cycle) * 10_000
        out = []
        for name, m in self.meta.items():
            if split == "heldout" and not m["frozen"]:
                continue
            items, err = sandbox.sample_family(self.code(name), range(base + 1, base + 1 + n_per_family), m["level"])
            if err:
                continue
            for it in items:
                rec = {"suite": f"factory:{name}", "id": f"{name}-{split}-c{cycle}-s{it['seed']}", "kind": it["kind"], "level": m["level"]}
                if it["kind"] == "code":
                    rec.update(prompt=it["prompt"] + "\n\nReply with the complete function in a single ```python code block.",
                               gold="", tests=it["tests"], entry_point=it["entry_point"])
                else:
                    rec.update(prompt=it["prompt"] + SUFFIX, gold=it["answer"].strip().replace(",", ""))
                out.append(rec)
        return out

    def update_levels(self, dev_records):
        """keep each family in the informative band: >85% correct -> harder, <25% -> easier"""
        by = {}
        for r in dev_records:
            if r["suite"].startswith("factory:"):
                by.setdefault(r["suite"][8:], []).append(r["correct"])
        changes = {}
        for name, v in by.items():
            if name not in self.meta or len(v) < 3:
                continue
            acc = float(np.mean(v)); m = self.meta[name]; m["dev_acc_history"] = (m["dev_acc_history"] + [round(acc, 2)])[-10:]
            if acc > 0.85 and m["level"] < 5:
                m["level"] += 1; changes[name] = f"acc {acc:.0%} -> level {m['level']}"
            elif acc < 0.25 and m["level"] > 1:
                m["level"] -= 1; changes[name] = f"acc {acc:.0%} -> level {m['level']}"
        self.save()
        return changes

    def summary(self):
        return {n: {"skill": m["skill"], "frozen": m["frozen"], "level": m["level"], "recent_dev_acc": m["dev_acc_history"][-3:]} for n, m in self.meta.items()}


FAMILY_SCHEMA = {"type": "object", "properties": {"families": {"type": "array", "items": {"type": "object", "properties": {
    "name": {"type": "string"}, "skill": {"type": "string"}, "code": {"type": "string"}}, "required": ["name", "skill", "code"]}}},
    "required": ["families"]}
REVIEW_SCHEMA = {"type": "object", "properties": {"wrong_ids": {"type": "array", "items": {"type": "string"}},
                                                  "ambiguous_ids": {"type": "array", "items": {"type": "string"}}, "comment": {"type": "string"}},
                 "required": ["wrong_ids", "ambiguous_ids", "comment"]}


def design(pilot, fac, focus, n_new, frozen, origin, evidence=None, journal=None):
    """ask the pilot for new generator families, validate them in the sandbox, have the pilot review samples"""
    brief = {"focus": focus, "n_new": n_new, "task": f"Write {n_new} new problem-generator families that test the focus. Each family should probe a "
             "DIFFERENT sub-skill the focus needs. Prefer families where the model is likely to fail for reasons you can name.",
             "contract": CONTRACT, "example_family": {"name": "unit_rate", "skill": BUILTIN["unit_rate"][0], "code": BUILTIN["unit_rate"][1]},
             "existing_families": fac.summary(), "evidence_from_model_failures": evidence or "none yet"}
    accepted, errors = [], {}
    for attempt in range(2):
        out, _ = pilot.ask("DESIGN_PROBLEMS", brief | ({"previous_errors": errors} if errors else {}), FAMILY_SCHEMA)
        errors = {}
        for f in out["families"][: n_new + 2]:
            e = validate_family(f["code"])
            if e:
                errors[f["name"]] = e
            else:
                accepted.append(f)
        if accepted:
            break
    kept = []
    for f in accepted[:n_new]:
        items, _ = sandbox.sample_family(f["code"], range(101, 105), 2)
        rev_items = [{"id": f"{f['name']}-{i}", "prompt": it["prompt"], "answer": it.get("answer", "(code tests) " + it.get("tests", "")[:400])} for i, it in enumerate(items)]
        rev, _ = pilot.ask("REVIEW_PROBLEMS", {"instruction": "Solve each problem yourself. List the ids whose stated answer is WRONG, and the ids "
                                               "whose wording is ambiguous. Be strict: a wrong answer key poisons every later measurement.",
                                               "items": rev_items}, REVIEW_SCHEMA)
        bad = set(rev["wrong_ids"]) | set(rev["ambiguous_ids"])
        if bad:
            errors[f["name"]] = f"review flagged {sorted(bad)}: {rev['comment'][:200]}"
            continue
        kept.append(fac.add(f["name"], f["skill"], f["code"], frozen, origin))
    if journal:
        journal("factory_design", origin=origin, frozen=frozen, kept=kept, rejected=errors)
    return kept

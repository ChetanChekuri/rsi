"""Problem formatting, answer extraction and grading, plus a pool of general problems (GSM8K + ARC-Challenge test)
used to check that an edit does not make the model worse at things outside the goal."""
import json, os, random, re

SUFFIX = "\n\nEnd your reply with a final line of the form 'Answer: <answer>'."
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def chat_prefix(user, think=True):
    """Qwen3.x chat format as ollama's qwen3.5 renderer produces it"""
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n" + ("<think>\n" if think else "<think>\n\n</think>\n\n")


def general_pool(exclude=()):
    """GSM8K + ARC-Challenge test problems in a fixed shuffled order (each cycle uses a new slice)"""
    out = []
    for i, l in enumerate(open(os.path.join(DATA, "gsm8k_test.jsonl"))):
        r = json.loads(l)
        out.append({"suite": "general:gsm8k", "id": f"gsm8k-{i}", "prompt": r["question"] + SUFFIX,
                    "gold": r["answer"].split("####")[-1].strip().replace(",", ""), "kind": "num"})
    for l in open(os.path.join(DATA, "arc_challenge_test.jsonl")):
        r = json.loads(l)
        ch = "\n".join(f"{a}. {t}" for a, t in zip(r["labels"], r["texts"]))
        out.append({"suite": "general:arc", "id": f"arc-{r['id']}", "prompt": f"{r['question']}\n\n{ch}\n\nAnswer with the letter of the correct option." + SUFFIX,
                    "gold": r["answer"].strip(), "kind": "choice"})
    out = [x for x in out if x["id"] not in set(exclude)]
    random.Random(1234).shuffle(out)
    return out


def extract(text):
    m = re.findall(r"Answer:\s*\**\s*([^\n]*)", text or "")
    if m:
        return m[-1].strip().strip("*$. ")
    tail = (text or "").strip().splitlines()[-1:] or [""]
    return tail[0]


def grade(item, pred):
    p = (pred or "").replace(",", "").replace("$", "").replace("\\", "").lower()
    p = re.sub(r"</?answer>", " ", p); g = str(item["gold"]).lower()
    if item["kind"] == "choice":
        m = re.findall(r"\b([a-e1-5])\b", p); return bool(m) and m[0] == g
    if item["kind"] == "word":
        return re.search(rf"\b{re.escape(g)}\b", p) is not None
    nums = re.findall(r"-?\d+(?:\.\d+)?", p)
    if not nums:
        return False
    try:
        return abs(float(nums[0]) - float(g)) < 1e-6 or (len(nums) <= 2 and abs(float(nums[-1]) - float(g)) < 1e-6)
    except ValueError:
        return False

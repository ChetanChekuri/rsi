"""Locally generated reasoning and instruction-following problems with exact answers."""
import random


def generate(seed, per_family=16):
    """Produce balanced families. Callers own the seed and decide who sees the answers."""
    rng = random.Random(seed)
    rows = []
    for i in range(per_family):
        a, b, c, d = (rng.randrange(11, 100) for _ in range(4))
        rows.append({"id": f"arithmetic-{i}", "family": "arithmetic",
                     "prompt": f"Compute {a} × {b} + {c} − {d}.", "answer": str(a * b + c - d)})
        boxes, size = rng.randrange(7, 80), rng.randrange(6, 70)
        sold = rng.randrange(2, boxes * size)
        returned = rng.randrange(1, sold)
        rows.append({"id": f"inventory-{i}", "family": "inventory",
                     "prompt": f"A shop starts with {boxes} boxes of {size} pens each. It sells {sold} pens. "
                               f"A customer returns {returned} pens to stock. How many pens are in stock now?",
                     "answer": str(boxes * size - sold + returned)})
        people = rng.sample(("Amir", "Ava", "Cora", "Diego", "Eli", "Inez", "Jia", "Noor", "Sam", "Zuri"), 5)
        clues = [f"{a} arrived before {b}." if rng.getrandbits(1) else f"{b} arrived after {a}."
                 for a, b in zip(people, people[1:])]
        rng.shuffle(clues)
        rank = rng.randrange(len(people))
        rows.append({"id": f"ordering-{i}", "family": "ordering",
                     "prompt": " ".join(clues) + f" Who arrived {('first', 'second', 'third', 'fourth', 'fifth')[rank]}?",
                     "answer": people[rank]})
        codes = [f"{x}-{rng.randrange(1000, 10000)}" for x in rng.sample("ABCDEFGHJKLMNPQR", 4)]
        colors = rng.sample(("blue", "green", "red", "yellow", "white", "purple"), 4)
        index = rng.randrange(4)
        rows.append({"id": f"lookup-{i}", "family": "lookup",
                     "prompt": ", ".join(f"{color} = {code}" for color, code in zip(colors, codes))
                               + f". Give the code for {colors[index]}.", "answer": codes[index]})
        values = [rng.randrange(-20, 30) for _ in range(6)]
        threshold = rng.randrange(-10, 20)
        rows.append({"id": f"code-{i}", "family": "code",
                     "prompt": f"In Python, values = {values}. What does sum(x * 2 for x in values if x > {threshold}) evaluate to?",
                     "answer": str(sum(x * 2 for x in values if x > threshold))})
        digits = [rng.randrange(10, 100) for _ in range(4)]
        rows.append({"id": f"format-{i}", "family": "format",
                     "prompt": f"Take {digits}. Keep only the even values, sort them descending, and join them with | and no spaces. "
                               "If there are none, give NONE.",
                     "answer": "|".join(map(str, sorted((x for x in digits if x % 2 == 0), reverse=True))) or "NONE"})
    rng.shuffle(rows)
    return rows


ANSWER_FORMAT = "\n\nEnd your reply with a line containing only Answer: followed by your answer."

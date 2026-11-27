"""Paired private goal evaluation plus per-family regression guards."""
import copy
import json
import math
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import goal_eval, llm, procedural, tests
from .tests import digest_for

STATE = "state/promotion.json"
VERSION = 2
PER_FAMILY = 16
OPTIONS = {"temperature": 0, "top_p": 1, "top_k": 1, "min_p": 0,
           "repeat_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0, "num_ctx": 2048}


def score(cfg, model, problem):
    messages = [{"role": "system", "content": "Solve the problem and follow its answer format."},
                {"role": "user", "content": problem["prompt"] + procedural.ANSWER_FORMAT}]
    reply = llm.chat(cfg, model, messages, think=False, num_predict=512, seed=0, options=OPTIONS, timeout=120, retries=2)
    content = reply["message"].get("content", "")
    return reply.get("done_reason") == "stop" and tests.committed_answer(content) is not None and bool(
        tests.default_grade(problem, {"content": content}))


def compare(problems, parent, candidate, goal_parent=(), goal_candidate=(), alpha=0.05):
    by_family = {p["family"]: {"fixed": 0, "broken": 0} for p in problems}
    for problem, before, after in zip(problems, parent, candidate, strict=True):
        by_family[problem["family"]]["fixed"] += int(after and not before)
        by_family[problem["family"]]["broken"] += int(before and not after)
    regressed = [family for family, row in by_family.items() if row["broken"] > row["fixed"]]
    pairs = list(zip(goal_parent, goal_candidate, strict=True))
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for pair in pairs for x in pair):
        raise ValueError("goal scores must be finite and between zero and one")
    wins = sum(b > a + 1e-9 for a, b in pairs)
    losses = sum(a > b + 1e-9 for a, b in pairs)
    p = tests.sign_p(wins, losses)
    mean_before = sum(goal_parent) / len(pairs) if pairs else 0
    mean_after = sum(goal_candidate) / len(pairs) if pairs else 0
    improved = bool(pairs) and wins > losses and mean_after > mean_before and p <= alpha
    return {"accepted": improved and not regressed, "n": len(problems),
            "parent_correct": sum(parent), "candidate_correct": sum(candidate),
            "by_family": by_family, "regressed_families": regressed,
            "goal": {"n": len(pairs), "parent_score": mean_before, "candidate_score": mean_after,
                     "fixed": wins, "broken": losses, "paired_p": p, "alpha": alpha},
            "reason": "passed" if improved and not regressed else
                      "regression guard failed" if regressed else "goal improvement was not established"}


def evaluate(loop, record, cycle):
    """Retries resume frozen tasks and completed outcomes; completed submissions cannot be rerolled."""
    cfg = loop.cfg
    if not cfg.get("sandbox_isolate", True) or cfg.get("sandbox_local", False) or cfg.get("sandbox_internet", False):
        raise ValueError("automatic promotion requires isolation with both sandbox network settings off")
    try:
        with open(loop.s.p(STATE), encoding="utf-8") as source:
            ledger = json.load(source)
    except FileNotFoundError:
        ledger = {"attempts": []}
    attempts = ledger["attempts"]
    parent, candidate = loop.tag, record["tag"]
    tags = tests.model_digests(cfg)
    pd, cd = digest_for(tags, parent), digest_for(tags, candidate)
    if not pd or not cd:
        raise llm.ServerDown("the evaluator could not verify both model digests; retry later")
    if pd != loop.gen.get("digest") or cd != record.get("digest"):
        raise ValueError("a model digest changed after it was recorded")
    if pd == cd:
        raise ValueError("the candidate has the same digest as its parent")
    previous = next((a for a in attempts if a["parent_digest"] == pd and a["candidate_digest"] == cd
                     and a.get("contract", {}).get("goal", cfg.get("goal")) == cfg.get("goal")), None)
    if previous and previous.get("state") == "finished" and previous.get("report", {}).get("accepted"):
        # Evaluation is durable before the separate lineage update. Replay an undelivered acceptance after a crash.
        activated = any(g.get("digest") == cd for g in getattr(loop, "lineage", {}).get("generations", []))
        if previous.get("contract", {}).get("goal") != cfg.get("goal"):
            raise ValueError("the goal changed after evaluation; cannot replay this acceptance")
        judge = previous.get("contract", {})
        if not previous.get("activated") and not activated:
            if digest_for(tags, judge.get("judge", "")) != judge.get("digest"):
                raise ValueError("the frozen evaluator digest changed after evaluation")
            return previous["report"]
    for attempt in attempts:
        if cycle in attempt.get("cycles", [attempt["cycle"]]) and not (
                attempt is previous and attempt["state"] in ("running", "retryable", "failed")):
            raise ValueError("the independent evaluator already used this cycle's submission")
    if previous and previous["state"] not in ("running", "retryable", "failed"):
        raise ValueError("this candidate already has an evaluation against this parent")
    frozen = goal_eval.contract(loop)
    attempt = previous
    if attempt is None:
        # Spend a bounded error budget across adaptive submissions for this goal/parent.
        number = 1 + sum(a["parent_digest"] == pd and a.get("contract", {}).get("goal") == frozen["goal"] for a in attempts)
        attempt = {"cycle": cycle, "cycles": [], "version": VERSION, "seed": secrets.randbits(128),
                   "parent": parent, "candidate": candidate, "parent_digest": pd, "candidate_digest": cd,
                   "alpha": 0.05 / (number * (number + 1)), "contract": copy.deepcopy(frozen),
                   "settings": {"options": OPTIONS, "goal_options": goal_eval.OPTIONS, "think": False,
                                "num_predict": 512, "goal_num_predict": 2048, "per_family": PER_FAMILY}}
        attempts.append(attempt)
    elif attempt.get("version") != VERSION:
        # Legacy failed attempts had no goal tasks. Preserve their seed while upgrading the incomplete evaluation.
        attempt.update(version=VERSION, contract=copy.deepcopy(frozen), alpha=0.025)
        for key in ("guards", "tasks", "progress", "outcomes"):
            attempt.pop(key, None)
    elif attempt["contract"] != frozen:
        raise ValueError("the goal or frozen evaluator changed; cannot resume this evaluation")
    if cycle not in attempt.setdefault("cycles", [attempt["cycle"]]):
        attempt["cycles"].append(cycle)
    attempt.update(state="running")
    attempt.pop("error", None)

    def save():
        loop.s.write(STATE, ledger)

    save()
    try:
        if "guards" not in attempt:
            attempt["guards"] = procedural.generate(attempt["seed"], PER_FAMILY)
            save()
        if "tasks" not in attempt:
            count = max(goal_eval.ITEMS, math.ceil(math.log2(2 / attempt["alpha"])) + 2)
            attempt["tasks"] = goal_eval.tasks(cfg, frozen, attempt["seed"], count=count)
            save()
        problems, tasks = attempt["guards"], attempt["tasks"]
        progress = attempt.setdefault("progress", {})
        workers = max(1, min(8, int(cfg.get("workers") or 1)))
        for index, model in enumerate((parent, candidate), 1):
            loop.status_activity("test", f"independent evaluation · model {index}/2")
            results = progress.setdefault(model, {"guards": {}, "answers": {}, "goals": {}})
            missing = [(i, problem) for i, problem in enumerate(problems) if str(i) not in results["guards"]]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(score, cfg, model, problem): str(i) for i, problem in missing}
                failure = None
                for future in as_completed(futures):
                    try:
                        results["guards"][futures[future]] = bool(future.result())
                        save()
                    except Exception as exc:
                        failure = exc
                if failure is not None:
                    raise failure
            for i, task in enumerate(tasks):
                key = str(i)
                if key not in results["answers"]:
                    results["answers"][key] = goal_eval.answer(cfg, model, task)
                    save()
            llm.unload(cfg, model)
        # Judge anonymous responses in seed-shuffled order after both models have answered.
        import random
        work = [(model, i) for model in (parent, candidate) for i in range(len(tasks))]
        random.Random(attempt["seed"]).shuffle(work)
        for model, i in work:
            results = progress[model]; key = str(i)
            if key not in results["goals"]:
                results["goals"][key] = goal_eval.grade(cfg, frozen, tasks[i], results["answers"][key])
                save()
        llm.unload(cfg, frozen["judge"])
        final_tags = tests.model_digests(cfg)
        expected = {parent: pd, candidate: cd, frozen["judge"]: frozen["digest"]}
        if not final_tags:
            raise llm.ServerDown("model digests temporarily unavailable; retry this evaluation")
        if any(digest_for(final_tags, model) != digest for model, digest in expected.items()):
            raise ValueError("model digests changed during the evaluation")
        guards = [[progress[m]["guards"][str(i)] for i in range(len(problems))] for m in (parent, candidate)]
        goals = [[progress[m]["goals"][str(i)] for i in range(len(tasks))] for m in (parent, candidate)]
        report = compare(problems, *guards, *goals, alpha=attempt["alpha"])
        attempt.update(state="finished", report=report)
    except llm.ServerDown as exc:
        attempt.update(state="retryable", error=str(exc)[:500])
        raise llm.ServerDown("independent evaluation interrupted; retry the same candidate to resume its saved evaluation") from None
    except Exception as exc:
        attempt.update(state="invalid", error=f"{type(exc).__name__}: {exc}"[:500])
        raise RuntimeError("independent evaluation failed validation; this submission was not promoted") from exc
    finally:
        save()
    return report


def activated(loop, digest):
    ledger = loop.s.read(STATE, {"attempts": []})
    for attempt in ledger["attempts"]:
        if attempt["candidate_digest"] == digest and attempt.get("report", {}).get("accepted"):
            attempt["activated"] = True
    loop.s.write(STATE, ledger)

import io
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gguf
import numpy as np

import rsi_lib
from rsi import directions, gguf_io, grade, grow, goal_eval, llm, netgate, network, procedural, promotion, sandbox, tests, workspace_files
from rsi.gguf_io import GGUF, q8_0
from rsi.loop import Loop, Session
from rsi.tools import Toolbox


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        session = Session(root)
        Path(session.p("state")).mkdir(parents=True)
        workspace = root / "workspace"
        workspace.mkdir()
        self.record = {"tag": "candidate", "digest": "candidate-digest"}
        session.write("state/models.json", [self.record])
        self.loop = SimpleNamespace(cfg={"ollama_url": "http://127.0.0.1:11434", "model": "parent", "goal": "Improve reasoning"}, s=session, workspace=str(workspace),
                                    tag="parent", gen={"digest": "parent-digest"}, probation=None,
                                    lineage={"current": 0}, note=Mock(), status_activity=Mock(), switch=Mock())
        capabilities = patch.object(llm, "model_caps", return_value={"caps": {"completion", "tools"}, "ctx": None})
        capabilities.start(); self.addCleanup(capabilities.stop)
        self.toolbox = Toolbox(self.loop, 5)
        self.tags = {"parent": "parent-digest", "candidate": "candidate-digest"}
        self.items = {p["prompt"] + procedural.ANSWER_FORMAT: p for p in procedural.generate(37, promotion.PER_FAMILY)}

        self.frozen = {"goal": "Improve reasoning", "judge": "parent", "digest": "parent-digest",
                       "criteria": ["correct", "complete", "follows instructions"]}
        for method, value in (("contract", self.frozen), ("tasks", [{"prompt": f"goal {i}", "reference": "answer"} for i in range(24)])):
            patcher = patch.object(goal_eval, method, return_value=value)
            patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(goal_eval, "answer", side_effect=lambda cfg, model, task: {"content": model, "done_reason": "stop"})
        patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(goal_eval, "grade", side_effect=lambda cfg, frozen, task, response: float(response["content"] == "candidate"))
        patcher.start(); self.addCleanup(patcher.stop)

    def reply(self, _, model, messages, **options):
        problem = self.items[messages[-1]["content"]]
        self.assertEqual(options["options"]["temperature"], 0)
        correct = model == "candidate" or problem["family"] == "lookup"
        return {"message": {"content": "Answer: " + (problem["answer"] if correct else "wrong")}, "done_reason": "stop"}

    def test_promotes_once_on_private_paired_improvement(self):
        with patch.object(tests, "model_digests", return_value=self.tags), patch.object(
                promotion.secrets, "randbits", return_value=37), patch.object(llm, "chat", side_effect=self.reply), \
                patch.object(llm, "unload"):
            result = self.toolbox.t_switch_model("candidate", "better")
            self.loop.switch.assert_called_once_with("candidate", "better", 5, "candidate-digest")
            with self.assertRaisesRegex(ValueError, "already used"):
                Toolbox(self.loop, 5).t_switch_model("candidate", "retry")
        self.assertIn("You are now candidate", result)
        receipt = self.loop.s.read(promotion.STATE)["attempts"][0]
        self.assertTrue(receipt["report"]["accepted"])
        self.assertEqual(receipt["report"]["n"], 96)
        self.assertEqual(receipt["seed"], 37)
        self.assertEqual(list(Path(self.loop.workspace).iterdir()), [])

    def test_rejects_regression_and_changed_digest(self):
        with patch.object(tests, "model_digests", return_value=self.tags), patch.object(
                llm, "chat", return_value={"message": {"content": "Answer: wrong"}, "done_reason": "stop"}), \
                patch.object(llm, "unload"), patch.object(goal_eval, "grade", return_value=0):
            self.assertIn("candidate rejected", self.toolbox.t_switch_model("candidate", "try"))
        self.loop.switch.assert_not_called()
        with patch.object(tests, "model_digests", return_value=dict(self.tags, candidate="replacement")):
            with self.assertRaisesRegex(ValueError, "digest changed"):
                Toolbox(self.loop, 6).t_switch_model("candidate", "try")

    def test_guard_regression_blocks_improvement(self):
        problems = list(self.items.values())
        guard = [p["family"] == "lookup" for p in problems]
        improved = [not p for p in guard]
        self.assertFalse(promotion.compare(problems, guard, improved)["accepted"])

    def test_training_job_can_use_pilot_curriculum_and_growth(self):
        self.loop.jobs = Mock()
        self.loop.gguf_of = Mock(return_value="/models/parent.gguf")
        self.toolbox.t_start_training("train", "out.gguf", problems_path="problems.jsonl", extra_neurons=32, layers=[2])
        _, command, cycle = self.loop.jobs.start.call_args.args
        argv = shlex.split(command)
        self.assertEqual(argv[:3], ["python", "-u", "-c"])
        self.assertEqual(cycle, 5)
        namespace = {}
        exec(argv[3].splitlines()[0], namespace)
        options = json.loads(namespace["ARGS"])
        self.assertEqual(Path(options["problems"]).name, "problems.jsonl")
        self.assertEqual(options["extra"], 32)
        self.assertEqual(options["base"], "/models/parent.gguf")
        compile(argv[3], "<training job>", "exec")

    def test_registration_stages_the_gguf_and_records_its_digest(self):
        ws = Path(self.loop.workspace)
        (ws / "new.gguf").write_bytes(b"GGUF fixture")
        self.loop.cfg.update(name="fixture", ollama_admin_url=self.loop.cfg["ollama_url"])
        self.loop.admin = "127.0.0.1:11434"
        self.loop.state = self.loop.s.p("state")
        self.loop.lineage["generations"] = []
        tag = "rsi-fixture:new"
        registered = dict(self.tags, **{tag: "new-digest"})

        def register(actual_tag, path, base, host, extra):
            self.assertEqual(actual_tag, tag)
            candidate = Path(path)
            self.assertFalse(candidate.is_relative_to(ws))
            self.assertEqual(candidate.read_bytes(), b"GGUF fixture")
            return "FROM new.gguf"

        with patch.object(tests, "model_digests", side_effect=[self.tags, registered, registered]), patch.object(
                gguf_io, "register", side_effect=register) as create, patch.object(llm, "model_caps", return_value={"caps": {"completion"}}):
            self.assertIn("registered as " + tag, self.toolbox.t_ollama_create("new", gguf_path="new.gguf"))
            self.assertIn("already exists", self.toolbox.t_ollama_create("new", gguf_path="new.gguf"))
        create.assert_called_once()
        self.assertEqual(self.loop.s.read("state/models.json")[-1]["digest"], "new-digest")

    def test_parent_cannot_complete_successor_probation(self):
        loop = Loop.__new__(Loop)
        loop.s, loop.workspace, loop.cfg = self.loop.s, self.loop.workspace, self.loop.cfg
        loop.cfg.update(goal="Improve reasoning", pilot_num_predict=128)
        loop.lineage = {"current": 0, "generations": [{"gen": 0, "tag": "parent", "digest": "parent-digest"}]}
        loop.probation = None
        loop.chat = []
        loop.jobs = Mock()
        loop.jobs.running.return_value = []
        loop.jobs.poll.return_value = []
        loop.reload = loop.write_context = loop.note = loop.status = Mock()
        loop.system_prompt = Mock(return_value="system prompt")
        loop._trim = lambda messages: messages
        reply = lambda *calls, content="": {"message": {"content": content, "tool_calls": [
            {"function": {"name": name, "arguments": args}} for name, args in calls]}}
        replies = [reply(("switch_model", {"model": "candidate"}), ("list_files", {})), reply(),
                   reply(("finish", {"summary": "done", "next_plans": ["try again"]}))]
        with patch.object(promotion, "evaluate", return_value={"accepted": True}), patch.object(llm, "chat", side_effect=replies):
            loop.cycle(5, None)
        self.assertEqual(loop.tag, "parent")
        self.assertEqual(loop.lineage["generations"][-1]["reverted"], "gave an empty reply")
        self.assertTrue(any("call skipped" in m.get("content", "") for m in loop.chat))


class HarnessTests(unittest.TestCase):
    def test_answer_grading(self):
        for answer, expected in (("$1,000", 1), ("1000.0", 1), ("1000 or 99", 0), ("1000 is wrong", 0),
                                 ("1000\nActually, 99", 0)):
            with self.subTest(answer=answer):
                self.assertEqual(tests.default_grade({"answer": "1000"}, {"content": "Answer: " + answer}), expected)

    def test_empty_and_duplicate_suites_fail(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "problems.jsonl"
            path.touch()
            with self.assertRaisesRegex(ValueError, "at least one problem"):
                tests.load_suite(d, path)
            row = json.dumps({"id": "p1", "prompt": "1+1?", "answer": "2"})
            path.write_text(row + "\n" + row + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                tests.load_suite(d, path)

    def test_custom_grader_reads_snapshot_and_returns_scores(self):
        with tempfile.TemporaryDirectory() as d:
            suite = Path(d) / "suite"
            suite.mkdir()
            (suite / "helper.py").write_text("VALUE = 0.75\n")
            self.addCleanup(sys.modules.pop, "helper", None)

            def mock_sandbox(ws, argv, **kw):
                with patch.object(sys, "path", sys.path.copy()):
                    grade.main(*argv[3:])
                return 0, ""

            with patch.object(sandbox, "run_in_workspace", side_effect=mock_sandbox):
                scores = tests.grade_all({"ollama_url": "fixture"}, d, str(suite),
                                         "from helper import VALUE\ndef grade(*args): return VALUE\n",
                                         [{"id": "a"}], [{"id": "a"}], None)
            self.assertEqual(scores, [0.75])

    def test_tied_auc_and_truncated_positions(self):
        self.assertEqual(directions._auc(np.array([0, 0, 0, 1, 1, 1]), np.ones(6)), 0.5)
        enc = lambda text: (list(map(ord, text)), [(i, i + 1) for i in range(len(text))])
        record = {"id": "p", "thinking": "First line.\nAnother line.\nWait late.", "gold": "42", "kind": "num"}
        _, captures, _, _ = directions.decision_points(SimpleNamespace(enc_offsets=enc), [record],
                                                       {"p": {"prompt": "P"}}, max_tokens=14, fmt=lambda p: p)
        self.assertEqual(captures, [])

    def test_incomplete_stream_is_an_error(self):
        with patch.object(network, "open_local", return_value=io.BytesIO(b'{"message":{"content":"partial"}}\n')):
            with self.assertRaisesRegex(llm.ServerDown, "completion record"):
                llm._stream(None, 1, {"slot": "fixture"}, "fixture")

    def test_offline_sandbox_mounts_only_the_inference_socket(self):
        with tempfile.TemporaryDirectory() as ws, patch.object(
                sandbox.netgate, "ensure", return_value={"ollama": "/test/ollama.sock", "proxy": "/test/proxy.sock"}):
            argv, _, _ = sandbox.command(ws, ["true"], internet=False)
        self.assertIn("/test/ollama.sock", argv)
        self.assertNotIn("/test/proxy.sock", argv)
        self.assertNotIn(sandbox.ROOT, argv)

    def test_inference_door_forwards_an_ordinary_request(self):
        caller, upstream = Mock(), Mock()
        caller.recv.return_value = b'POST /api/chat HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}'
        response = upstream.getresponse.return_value
        response.status, response.reason = 200, "OK"
        response.getheaders.return_value = [("Content-Type", "application/x-ndjson")]
        response.read1.side_effect = [b'{"done":true}\n', b'']
        with patch.object(netgate.http.client, "HTTPConnection", return_value=upstream):
            netgate._ollama(caller, "http://localhost:11434")
        upstream.request.assert_called_once_with("POST", "/api/chat", body=b"{}", headers={"content-type": "application/json"})
        caller.sendall.assert_any_call(b'{"done":true}\n')

    def test_workspace_files_reject_links(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            item = root / "original"
            item.write_text("unchanged")
            alias = root / "alias"
            alias.symlink_to(item)
            with self.assertRaises(OSError):
                workspace_files.write(root, alias, "changed")
            self.assertEqual(item.read_text(), "unchanged")


class GrowthTests(unittest.TestCase):
    def test_widening_k_quant_preserves_dequantized_weights(self):
        with tempfile.TemporaryDirectory() as d:
            base, wide = (str(Path(d) / name) for name in ("base.gguf", "wide.gguf"))
            writer = gguf.GGUFWriter(base, "llama")
            writer.add_block_count(1)
            writer.add_feed_forward_length(256)
            rng = np.random.default_rng(12)
            packed = rng.integers(0, 255, size=(32, 210), dtype=np.uint8)
            packed[:, -2:] = np.array([0.01], dtype=np.float16).view(np.uint8)
            name = "blk.0.ffn_down.weight"
            writer.add_tensor(name, packed, raw_dtype=gguf.GGMLQuantizationType.Q6_K)
            for part in ("gate", "up"):
                writer.add_tensor(f"blk.0.ffn_{part}.weight", np.ones((256, 32), np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            grow.widen_mlp(base, wide, extra=32)
            grown = GGUF(wide)
            self.assertEqual(grown.T[name].tensor_type.name, "F32")
            np.testing.assert_array_equal(grown.deq(name)[:, :256], GGUF(base).deq(name))

    def test_growth_and_exact_update_freeze_existing_entries(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base, wide, deeper, trained = map(lambda n: str(root / (n + ".gguf")), ("base", "wide", "deeper", "trained"))
            name = "blk.0.ffn_down.weight"
            writer = gguf.GGUFWriter(base, "llama")
            writer.add_block_count(1)
            writer.add_feed_forward_length(32)
            g = np.random.default_rng(42)
            for part in ("down", "gate", "up"):
                writer.add_tensor(f"blk.0.ffn_{part}.weight", q8_0(g.normal(size=(32, 32))),
                                  raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor("blk.0.attn_output.bias", np.ones(32, np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            original = GGUF(base).deq(name)
            spec = grow.widen_mlp(base, wide, extra=32)[name]
            np.testing.assert_array_equal(GGUF(wide).deq(name)[:, :32], original)
            self.assertEqual(np.count_nonzero(GGUF(wide).deq(name)[:, 32:]), 0)
            measurement = {"base": wide, "exact": {name: (spec, np.ones((32, 32), np.float32))}}
            rsi_lib.gradient_step(measurement, trained, kind="exact")
            np.testing.assert_array_equal(GGUF(trained).deq(name)[:, :32], original)
            self.assertEqual(np.count_nonzero(GGUF(trained).deq(name)[:, 32:]), 32 * 32)
            grow.add_layers(base, deeper, after=0)
            self.assertEqual(np.count_nonzero(GGUF(deeper).deq("blk.1.attn_output.bias")), 0)


if __name__ == "__main__":
    unittest.main()

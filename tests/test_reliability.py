import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from rsi import conversation, goal_eval, gradstep, journal, llm, network, netgate, procedural, promotion, research, sandbox, service, settings, tests
from rsi.arch_qwen35 import Qwen35Stream
from rsi.loop import Loop, Session


class NetworkTests(unittest.TestCase):
    def setUp(self):
        research._paused.clear(); research._bad.clear()

    def test_tinyfish_errors_never_fall_back(self):
        with patch.object(research, 'key', return_value='fixture'), patch.object(research, '_get', side_effect=OSError('outage')) as get:
            for fn, arg in ((research.web_search, 'query'), (research.arxiv_search, 'query'), (research.fetch_page, 'https://example.org/page')):
                with self.assertRaises(research.ResearchUnavailable):
                    fn(arg)
            self.assertEqual(len(get.call_args_list), 3)
            self.assertTrue(all('tinyfish.ai' in call.args[0] for call in get.call_args_list))

    def test_missing_key_and_quota_do_not_make_requests(self):
        with patch.object(research, '_get') as get, patch.object(research, 'key', return_value=None):
            with self.assertRaises(research.ResearchUnavailable):
                research.web_search('query')
            get.assert_not_called()
        research._paused['tinyfish'] = float('inf')
        with patch.object(research, '_get') as get, patch.object(research, 'key', return_value='fixture'):
            with self.assertRaises(research.ResearchUnavailable):
                research.web_search('query')
            get.assert_not_called()

    def test_empty_results_and_arxiv_use_only_tinyfish(self):
        with patch.object(research, 'key', return_value='fixture'), patch.object(research, '_get', return_value=('application/json', b'{"results": []}')) as get:
            self.assertEqual(research.web_search('query'), [])
            self.assertEqual(research.arxiv_search('query'), [])
            self.assertEqual(get.call_count, 2)
            self.assertIn('site%3Aarxiv.org', get.call_args.args[0])

    def test_provider_allowlist_and_redirects(self):
        with patch.object(urllib.request, 'build_opener') as opener:
            for url in ('https://example.org', 'http://api.search.tinyfish.ai', 'https://api.search.tinyfish.ai.evil.test', 'https://user@api.fetch.tinyfish.ai'):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    research._get(url)
            opener.assert_not_called()
        with self.assertRaises(ValueError):
            network.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.org')

    def test_code_cannot_enable_general_network(self):
        with tempfile.TemporaryDirectory() as ws:
            for option in ({'internet': True}, {'local': True}, {'isolate': False}):
                with self.subTest(option=option), self.assertRaises(ValueError):
                    sandbox.command(ws, ['true'], **option)

    def test_local_inference_policy_and_no_proxy(self):
        for url in ('https://ollama.com', 'http://192.168.1.10:11434', 'http://localhost@external.test', 'file:///tmp/model'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                network.local_url(url)
        self.assertEqual(network.local_url('http://[::1]:11434/'), 'http://[::1]:11434')
        with patch.object(urllib.request, 'build_opener') as build:
            network.open_local('http://127.0.0.1:11434/api/tags')
            self.assertEqual(build.call_args.args[0].proxies, {})

    def test_cloud_model_rejected_before_generation(self):
        llm._CAPS.clear()
        data = io.BytesIO(json.dumps({'remote_host': 'https://cloud.example', 'capabilities': ['completion']}).encode())
        with patch.object(network, 'open_local', return_value=data) as call:
            with self.assertRaisesRegex(ValueError, 'cloud-backed'):
                llm.chat({'ollama_url': 'http://127.0.0.1:11434'}, 'cloud', [])
            self.assertEqual(call.call_count, 1)
            self.assertTrue(call.call_args.args[0].full_url.endswith('/api/show'))

    def test_nonstream_incomplete_response_is_not_a_score(self):
        body = io.BytesIO(b'{"message":{"content":"partial"},"done":false}')
        with patch.object(llm, 'model_caps', return_value={'caps': {'completion'}, 'ctx': None}), patch.object(
                network, 'open_local', return_value=body), self.assertRaisesRegex(llm.ServerDown, 'incomplete'):
            llm.chat({'ollama_url': 'http://127.0.0.1:11434'}, 'local', [], retries=1)

    def test_option_overrides_cannot_exceed_model_context(self):
        body = io.BytesIO(b'{"message":{"content":"answer"},"done":true,"done_reason":"stop"}')
        with patch.object(llm, 'model_caps', return_value={'caps': {'completion'}, 'ctx': 2048}), patch.object(
                network, 'open_local', return_value=body) as request:
            llm.chat({'ollama_url': 'http://127.0.0.1:11434'}, 'local', [], options={'num_ctx': 16000}, num_predict=8000)
        sent = json.loads(request.call_args.args[0].data)
        self.assertEqual(sent['options']['num_ctx'], 2048)
        self.assertEqual(sent['options']['num_predict'], 1024)

    def test_inference_door_rejects_cloud_model_with_uppercase_field(self):
        caller = Mock()
        body = b'{"Model":"remote"}'
        caller.recv.return_value = b'POST /api/chat HTTP/1.1\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body
        with patch.object(llm, 'model_caps', side_effect=ValueError('cloud model')) as verify, patch.object(
                netgate.http.client, 'HTTPConnection') as forward:
            netgate._ollama(caller, 'http://127.0.0.1:11434')
            self.assertEqual(verify.call_args.args[1], 'remote')
            forward.assert_not_called()
            self.assertIn(b'403 Forbidden', caller.sendall.call_args.args[0])

    def test_old_settings_cannot_reopen_network(self):
        cfg = settings.effective(dict(settings.defaults(), sandbox_local=True, sandbox_internet=True, sandbox_isolate=False), {})
        self.assertEqual([cfg[k] for k in ('sandbox_local', 'sandbox_internet', 'sandbox_isolate')], [False, False, True])


class JournalTests(unittest.TestCase):
    def test_interrupted_tail_is_readable_and_repaired_on_append(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.jsonl'
            path.write_bytes(b'{"cycle":0}\n{"cycle":1,"summary":"broken')
            self.assertEqual(journal.read(path), [{'cycle': 0}])
            journal.append(path, {'cycle': 1, 'summary': 'recovered'})
            self.assertEqual(journal.read(path), [{'cycle': 0}, {'cycle': 1, 'summary': 'recovered'}])
            backups = list(Path(directory).glob('*.interrupted-*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), b'{"cycle":1,"summary":"broken')

    def test_valid_unterminated_record_kept_and_middle_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.jsonl'
            path.write_text('{"cycle":0}')
            journal.append(path, {'cycle': 1})
            self.assertEqual(len(journal.read(path)), 2)
            path.write_text('{bad}\n{"cycle":1}\n')
            with self.assertRaises(ValueError):
                journal.read(path)
            with self.assertRaises(ValueError):
                journal.append(path, {'cycle': 2})

    def test_concurrent_writers_do_not_lose_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.jsonl'
            with ThreadPoolExecutor(4) as executor:
                list(executor.map(lambda i: journal.append(path, {'i': i}), range(40)))
            self.assertEqual({r['i'] for r in journal.read(path)}, set(range(40)))


class ConversationTests(unittest.TestCase):
    def messages(self):
        messages = [{'role': 'user', 'content': 'Current cycle: train a candidate.'}]
        for i in range(20):
            messages += [{'role': 'assistant', 'content': '', 'tool_calls': [
                {'function': {'name': 'run_python', 'arguments': {'code': f'print({i})'}}}]},
                {'role': 'tool', 'tool_name': 'run_python', 'content': f'Result {i}: ' + 'x' * 1500}]
        return messages

    def test_keeps_recent_results_and_their_calls(self):
        original = self.messages()
        result = conversation.trim(original, 6000)
        self.assertTrue(any(m['role'] == 'tool' and m['content'].startswith('Result 19:') for m in result))
        self.assertLess(sum(len(json.dumps(m, ensure_ascii=False)) for m in result), 6000)
        for i, message in enumerate(result):
            if message['role'] == 'tool':
                self.assertTrue(result[i-1].get('tool_calls'))
        self.assertEqual(len(original), 41)

    def test_oversized_last_exchange_retains_result_in_summary(self):
        messages = self.messages()
        messages[-2]['tool_calls'][0]['function']['arguments']['code'] = 'x' * 20000
        messages[-1]['content'] = 'Critical latest result: training succeeded'
        result = conversation.trim(messages, 5000)
        self.assertIn('training succeeded', json.dumps(result))
        self.assertLess(sum(len(json.dumps(m, ensure_ascii=False)) for m in result), 5000)

    def test_full_playbook_leaves_room_for_recent_work_and_reply(self):
        loop = Loop.__new__(Loop)
        loop.probation = None
        loop.lineage = {'current': 0, 'generations': [{'gen': 0, 'tag': 'parent'}]}
        loop.hw = {'label': 'test CPU'}
        loop.gguf_of = lambda tag: '/models/test.gguf'
        loop.s = SimpleNamespace(history=lambda: [])
        for ctx in (8192, 32768):
            with self.subTest(ctx=ctx):
                loop.cfg = dict(settings.defaults(), model='parent', goal='Improve reasoning', master_tag='parent', num_ctx=ctx)
                with patch.object(llm, 'model_caps', return_value={'ctx': ctx}):
                    loop._fit_context()
                messages = loop._trim(self.messages())
                from rsi.tools import tool_list
                total = (len(loop.system_prompt()) + len(json.dumps(tool_list(loop.cfg)))
                         + sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)) // 3
                self.assertLess(total + loop.cfg['pilot_num_predict'] + 512, ctx)
                self.assertIn('Result 19:', json.dumps(messages))



class SmallStream:
    L, D, bos, eps = 1, 2, [], 1e-6
    lin, rms = Qwen35Stream.lin, Qwen35Stream.rms
    def __init__(self, dtype, device='cpu'):
        self.dev = torch.device(device)
        self.g = SimpleNamespace(deq=lambda name: np.ones(2, dtype=np.float32))
        self.weight = torch.tensor([[0.2, -0.3], [0.4, 0.2]], dtype=dtype, device=self.dev)
    def enc(self, text):
        return [0] if text == 'p' else [0, 1]
    def _lm_chunks(self, **kw):
        yield 0, 3, torch.tensor([[0.1, 0.2], [0.3, -0.4], [-0.2, 0.1]], device=self.dev)
    def _embed(self, seqs):
        return [torch.tensor([[0.25, 0.5]] * len(s), device=self.dev) for s in seqs]
    def _layer(self, index):
        return {'blk.0.ffn_down.weight': self.weight}
    def block(self, index, x, weights):
        return x + self.lin(x, weights['blk.0.ffn_down.weight'])


class GradientTests(unittest.TestCase):
    def test_half_precision_gradient_matches_float_reference(self):
        measured = []
        for dtype in (torch.float32, torch.float16):
            result = gradstep.measure(SmallStream(dtype), [('p', 'a')], layers=[0], rank=1, sketch=2)
            measured.append(result['gain_grads'])
        np.testing.assert_allclose(measured[0], measured[1], rtol=0.01, atol=0.001)
        self.assertTrue(np.isfinite(measured[1]).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable')
    def test_cuda_gradient(self):
        result = gradstep.measure(SmallStream(torch.float16, 'cuda'), [('p', 'a')], layers=[0], rank=1, sketch=2)
        self.assertTrue(np.isfinite(result['gain_grads']).all())


class ServiceTests(unittest.TestCase):
    def test_persistent_service_enabled_with_linger_and_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / '.rsi'; state.mkdir()
            with patch.object(service.shutil, 'which', return_value='/bin/fixture'), patch.object(service, '_run', return_value='no') as run:
                service.start(directory, {'memory_cap': '4G', 'cpu_quota': '200%'})
                commands = [call.args[0] for call in run.call_args_list]
                self.assertIn(['loginctl', '--no-ask-password', 'enable-linger', str(os.getuid())], commands)
                self.assertIn(['systemctl', '--user', 'enable', service.name(directory)], commands)
                self.assertIn(['systemctl', '--user', 'start', service.name(directory)], commands)
                unit = (state / service.name(directory)).read_text()
                for value in ('Restart=always', 'WantedBy=default.target', 'KillMode=control-group', 'ConditionPathExists=!', 'MemoryMax=4G'):
                    self.assertIn(value, unit)
                service.stop(directory)
                run.assert_called_with(['systemctl', '--user', 'stop', service.name(directory)])

    @unittest.skipUnless(shutil.which('systemd-analyze'), 'systemd validator unavailable')
    def test_unit_validates_with_spaces_in_paths(self):
        with tempfile.TemporaryDirectory(prefix='rsi service ') as directory:
            path = Path(directory) / service.name(directory)
            path.write_text(service.unit(directory, {}))
            result = subprocess.run(['systemd-analyze', 'verify', '--man=no', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, '')

    def test_restart_preserves_previously_saved_api_key(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / '.rsi'; state.mkdir()
            with patch.object(service.shutil, 'which', return_value='/bin/fixture'), patch.object(service, '_run', return_value='yes'):
                with patch.dict(os.environ, {'TINYFISH_API_KEY': 'fixture-key'}, clear=True):
                    service.start(directory, {})
                with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}, clear=True):
                    service.start(directory, {})
            self.assertIn('fixture-key', (state / 'service.env').read_text())
            self.assertEqual((state / 'service.env').stat().st_mode & 0o777, 0o600)

    def test_failed_service_setup_is_reported(self):
        with patch.object(service.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stderr='permission denied', stdout='')):
            with self.assertRaisesRegex(RuntimeError, 'permission denied'):
                service._run(['systemctl', '--user', 'start', 'fixture.service'])

    def test_startup_retries_before_entering_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Session(directory); Path(session.dir).mkdir()
            loop = Loop.__new__(Loop)
            loop.s = session; loop.cycle_k = 0; loop.cfg = {}
            loop.setup = Mock(side_effect=[RuntimeError('temporary startup failure'), None])
            loop.note = loop.status = Mock(); loop._inbox = lambda: []
            loop.cycle = Mock(side_effect=KeyboardInterrupt)
            with patch('rsi.loop.Jobs'), patch('rsi.loop.signal.signal'), patch('rsi.loop.time.sleep') as sleep, patch.object(goal_eval, 'contract'):
                with self.assertRaises(KeyboardInterrupt):
                    loop._run_locked()
            self.assertEqual(loop.setup.call_count, 2)
            sleep.assert_called_once_with(30)


class EvaluationTests(unittest.TestCase):
    def test_a_family_regression_blocks_even_strong_goal_improvement(self):
        problems = procedural.generate(1)
        parent = [p['family'] in ('ordering', 'lookup') for p in problems]
        candidate = [p['family'] != 'ordering' for p in problems]
        result = promotion.compare(problems, parent, candidate, [0] * 24, [1] * 24)
        self.assertFalse(result['accepted'])
        self.assertIn('ordering', result['regressed_families'])

    def test_saturated_guards_allow_real_goal_improvement(self):
        problems = procedural.generate(1)
        perfect = [True] * len(problems)
        self.assertTrue(promotion.compare(problems, perfect, perfect, [0] * 24, [1] * 24)['accepted'])
        self.assertFalse(promotion.compare(problems, perfect, perfect, [1] * 24, [1] * 24)['accepted'])
        self.assertFalse(promotion.compare(problems, perfect, perfect)['accepted'])

    def test_evaluation_resumes_saved_tasks_and_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Session(directory); Path(session.p('state')).mkdir(parents=True)
            loop = SimpleNamespace(s=session, cfg={'goal': 'code'}, tag='parent', gen={'digest': 'pd'}, status_activity=Mock())
            record = {'tag': 'candidate', 'digest': 'cd'}
            frozen = {'goal': 'code', 'judge': 'judge', 'digest': 'jd', 'criteria': ['correct']}
            rows = [{'prompt': f'goal {i}', 'reference': 'answer'} for i in range(24)]
            cfg = {'parent': 'pd', 'candidate': 'cd', 'judge': 'jd'}
            completed = []
            def answer(config, model, task):
                if model == 'parent' and task['prompt'] == 'goal 1' and not completed:
                    completed.append(True)
                    raise llm.ServerDown('temporary outage')
                return {'content': model, 'done_reason': 'stop'}
            with patch.object(tests, 'model_digests', return_value=cfg), patch.object(goal_eval, 'contract', return_value=frozen), \
                 patch.object(goal_eval, 'tasks', return_value=rows) as make_tasks, patch.object(promotion, 'score', return_value=True) as guard, \
                 patch.object(goal_eval, 'answer', side_effect=answer) as generate, \
                 patch.object(goal_eval, 'grade', side_effect=lambda cfg, frozen, task, response: float(response['content'] == 'candidate')), \
                 patch.object(llm, 'unload'), patch.object(promotion.secrets, 'randbits', return_value=37) as random:
                with self.assertRaises(llm.ServerDown):
                    promotion.evaluate(loop, record, 1)
                saved = session.read(promotion.STATE)['attempts'][0]
                self.assertEqual(saved['state'], 'retryable')
                self.assertEqual(saved['progress']['parent']['answers']['0']['content'], 'parent')
                result = promotion.evaluate(loop, record, 2)
                self.assertTrue(result['accepted'])
                self.assertEqual(random.call_count, 1)
                self.assertEqual(make_tasks.call_count, 1)
                self.assertEqual(guard.call_count, 192)
                first = [call for call in generate.call_args_list if call.args[1] == 'parent' and call.args[2]['prompt'] == 'goal 0']
                self.assertEqual(len(first), 1)
                # A crash between acceptance and handoff can replay the durable result.
                self.assertEqual(promotion.evaluate(loop, record, 2), result)
                promotion.activated(loop, 'cd')
                with self.assertRaisesRegex(ValueError, 'already has an evaluation'):
                    promotion.evaluate(loop, record, 3)

    def test_goal_contract_frozen_and_judge_digest_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Session(directory); Path(session.p('state')).mkdir(parents=True)
            loop = SimpleNamespace(s=session, cfg={'goal': 'write correct Python', 'model': 'original'})
            rubric = {'criteria': ['correct output', 'handles edge cases', 'valid Python']}
            with patch.object(tests, 'model_digests', return_value={'original': 'digest'}), patch.object(goal_eval, '_json', return_value=rubric) as make:
                first = goal_eval.contract(loop)
                self.assertEqual(first, goal_eval.contract(loop))
                self.assertEqual(make.call_count, 1)
            with patch.object(tests, 'model_digests', return_value={'original': 'changed'}), self.assertRaisesRegex(ValueError, 'digest changed'):
                goal_eval.contract(loop)

    def test_judge_cannot_omit_criteria_or_change_scale(self):
        frozen = {'judge': 'judge', 'criteria': ['one', 'two', 'three']}
        problem = {'prompt': 'task', 'reference': 'reference'}
        response = {'content': 'answer', 'done_reason': 'stop'}
        for result in ({'criteria': [{'id': 0, 'met': True}]}, {'criteria': [{'id': 0, 'met': True}] * 3}):
            with patch.object(goal_eval, '_json', return_value=result), self.assertRaises(llm.ServerDown):
                goal_eval.grade({}, frozen, problem, response)

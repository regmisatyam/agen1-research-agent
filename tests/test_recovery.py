"""Regression tests for the failures seen in runs/toon-json, using deterministic models."""
import asyncio
import base64
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

from agen1 import http_worker
from agen1.config import Config
from agen1.engine import Engine, excerpt_note, key_terms
from agen1.network import HTTP, HTTPError, LLM, extract_json, parse_actions, strip_reasoning
from agen1.storage import Journal, Store
from agen1.tools import Tools


TOPIC = 'How can TOON improve token efficiency compared to JSON in LLM applications?'
PAGE = 'TOON improves token efficiency compared to JSON in LLM prompts according to the benchmark. Unrelated footer text appears here for the reader.'


def reply(content, finish_reason='stop', **message):
    return {'choices': [{'message': {'content': content, **message}, 'finish_reason': finish_reason}],
            'usage': {'total_tokens': 100}}


class ParsingTests(unittest.TestCase):
    def test_exact_replies_that_failed_in_the_toon_run(self):
        flat = parse_actions('\n\n{"tool":"search_web","query":"TOON specification"}')
        self.assertEqual(flat, [{'tool': 'search_web', 'args': {'query': 'TOON specification'}}])
        fenced_flat = parse_actions('\n\n```json\n{"tool":"search_web","query":"TOON"}\n```')
        self.assertEqual(fenced_flat[0]['args'], {'query': 'TOON'})
        batch = parse_actions('\n\n```json\n[\n  {"tool": "fetch_page", "args": {"url": "https://a.invalid"}},\n  {"tool": "search_web", "args": {"query": "q"}}\n]\n```')
        self.assertEqual([a['tool'] for a in batch], ['fetch_page', 'search_web'])

    def test_other_common_shapes(self):
        self.assertEqual(parse_actions('<think>maybe {"tool":"x"}</think>{"tool":"finish","args":{}}')[0]['tool'], 'finish')
        self.assertEqual(parse_actions('I will search now: {"tool":"search_web","args":{"query":"q"}} done')[0]['args'], {'query': 'q'})
        self.assertEqual(parse_actions('<tool_call>{"name":"search_web","arguments":{"query":"q"}}</tool_call>')[0]['tool'], 'search_web')
        self.assertEqual(parse_actions('{"tool":"search_web","args":"{\\"query\\":\\"q\\"}"}')[0]['args'], {'query': 'q'})
        self.assertEqual(parse_actions('See [1] and {"note":1}. {"tool":"finish"}')[0]['tool'], 'finish')

    def test_invalid_replies_are_rejected_with_the_format(self):
        for text in ('no json here', '{"tool":1}', '[1,2]', '{"tool":"x","args":[1]}', '{"tool":"x","args":"not json"}', '{"tool":"search_web","args":{"query":'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'search_web'):
                parse_actions(text)

    def test_reasoning_removed_even_when_cut_off(self):
        self.assertEqual(strip_reasoning('<think>long</think>\n\nanswer'), 'answer')
        self.assertEqual(strip_reasoning('<think>never closed'), '')
        self.assertEqual(strip_reasoning(None), '')
        self.assertEqual(extract_json('```json\n{"assignments":["a"]}\n```'), {'assignments': ['a']})


class Fixture(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.config = Config(TOPIC, self.directory, hours=1 / 60, workers=1, supervisor_interval=.02,
                             request_timeout=1, demo=True, max_output_tokens=100)
        self.journal = Journal(self.directory, quiet=True)
        self.store = Store(self.directory, TOPIC, self.journal)

    def build(self, http):
        tools = Tools(self.config, self.store, self.journal, http)
        engine = Engine(self.config, LLM(self.config, self.journal, http), tools, self.store, self.journal)
        engine.pause = AsyncMock()
        return engine

    def events(self, kind):
        return [e for e in map(json.loads, self.journal.path.read_text().splitlines()) if e['kind'] == kind]


class Transport:
    """Answers search and page requests; `model` decides each chat reply."""
    def __init__(self, model):
        self.model = model
        self.urls = []

    async def json(self, url, **kw):
        self.urls.append(url)
        if url.endswith('/chat/completions'):
            return self.model(kw['body'])
        if 'semanticscholar' in url:
            return {'data': [{'paperId': 'p1', 'title': 'TOON paper', 'url': 'https://paper.invalid', 'abstract': PAGE, 'authors': []}]}
        return {'web': {'results': [{'title': 'TOON <strong>page</strong>', 'url': 'https://page.invalid', 'description': 'A snippet about TOON &amp; JSON'}]}}

    async def request(self, url, **kw):
        return ('<html><body><p>' + PAGE + '</p></body></html>').encode()


class EmptyReplyTests(Fixture):
    async def test_reasoning_overflow_retries_without_reasoning_and_stays_off(self):
        replies = [reply(None, 'length', reasoning_content='thinking' * 50), reply('answer'), reply('later')]
        bodies = []
        async def serve(url, **kw):
            bodies.append(kw['body'])
            return replies.pop(0)
        http = Transport(None)
        http.json = serve
        llm = LLM(self.config, self.journal, http)
        self.assertEqual(await llm.chat('a', [{'role': 'user', 'content': 'q'}]), 'answer')
        self.assertNotIn('chat_template_kwargs', bodies[0])
        self.assertEqual(bodies[1]['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual(bodies[1]['max_tokens'], 200)
        self.assertEqual(llm.tokens, 200, 'tokens spent on the rejected reply must count')
        self.assertIn('reasoning_chars=400', self.events('llm_retry')[0]['summary'])
        await llm.chat('a', [{'role': 'user', 'content': 'q'}])
        self.assertEqual(bodies[2]['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual(bodies[2]['max_tokens'], 100)

    async def test_server_rejecting_template_kwargs_still_gets_answers(self):
        bodies = []
        async def serve(url, **kw):
            bodies.append(kw['body'])
            if len(bodies) == 1:
                return reply('', 'length')
            if 'chat_template_kwargs' in kw['body']:
                raise HTTPError('HTTP 400 from model', False)
            return reply('answer')
        http = Transport(None)
        http.json = serve
        llm = LLM(self.config, self.journal, http)
        self.assertEqual(await llm.chat('a', [{'role': 'user', 'content': 'q'}]), 'answer')
        self.assertFalse(llm.template_kwargs_supported)

    async def test_timeout_retries_without_reasoning(self):
        for message in ('Request to model exceeded 60s', 'Connection failed to model: TimeoutError'):
            with self.subTest(message=message):
                bodies = []
                async def serve(url, **kw):
                    bodies.append(kw['body'])
                    if len(bodies) == 1:
                        raise HTTPError(message, True)
                    return reply('answer')
                http = Transport(None)
                http.json = serve
                llm = LLM(self.config, self.journal, http)
                with patch('agen1.network.asyncio.sleep', new=AsyncMock()):
                    self.assertEqual(await llm.chat('a', [{'role': 'user', 'content': 'q'}]), 'answer')
                self.assertNotIn('chat_template_kwargs', bodies[0])
                self.assertEqual(bodies[1]['chat_template_kwargs'], {'enable_thinking': False})

    async def test_other_failures_leave_reasoning_on(self):
        calls = []
        async def serve(url, **kw):
            calls.append(1)
            if len(calls) == 1:
                raise HTTPError('HTTP 429 from model', True)
            return reply('answer')
        http = Transport(None)
        http.json = serve
        llm = LLM(self.config, self.journal, http)
        with patch('agen1.network.asyncio.sleep', new=AsyncMock()):
            await llm.chat('a', [{'role': 'user', 'content': 'q'}])
        self.assertFalse(llm.disable_thinking)

    async def test_no_reasoning_option_applies_from_the_first_request(self):
        self.config.no_reasoning = True
        bodies = []
        def model(body):
            bodies.append(body)
            return reply('answer')
        await LLM(self.config, self.journal, Transport(model)).chat('a', [{'role': 'user', 'content': 'q'}])
        self.assertEqual(bodies[0]['chat_template_kwargs'], {'enable_thinking': False})

    async def test_persistently_empty_reply_reports_the_diagnosis(self):
        http = Transport(lambda body: reply('', 'length'))
        llm = LLM(self.config, self.journal, http)
        with self.assertRaisesRegex(ValueError, 'finish_reason=length'):
            await llm.chat('a', [{'role': 'user', 'content': 'q'}])
        self.assertEqual(llm.reserved, 0)

    async def test_truncated_paper_is_labelled_and_gets_more_room(self):
        bodies = []
        def model(body):
            bodies.append(body)
            return reply('# Draft', 'length')
        llm = LLM(self.config, self.journal, Transport(model))
        answer = await llm.chat('writer', [{'role': 'user', 'content': 'q'}], final=True)
        self.assertIn('cut off', answer)
        self.assertEqual(bodies[0]['max_tokens'], 4096)


class ToolTests(Fixture):
    async def test_missing_arguments_explain_the_correct_call(self):
        tools = Tools(self.config, self.store, self.journal, Transport(None))
        with self.assertRaisesRegex(ValueError, r'search_web requires query.*"args"'):
            await tools.execute('a', 'search_web', {})
        with self.assertRaisesRegex(ValueError, 'note requires citations'):
            await tools.execute('a', 'note', {'text': 'x'})

    async def test_unknown_arguments_are_ignored(self):
        tools = Tools(self.config, self.store, self.journal, Transport(None))
        self.assertTrue((await tools.execute('a', 'search_web', {'query': 'q', 'count': 3}))['progress'])

    async def test_single_string_citation_is_accepted(self):
        tools = Tools(self.config, self.store, self.journal, Transport(None))
        sid, _ = self.store.source('T', 'https://t.invalid', 'text', 'brave')
        self.assertTrue((await tools.execute('a', 'note', {'text': 'A finding', 'citations': sid}))['saved'])

    async def test_paper_search_works_without_key_and_falls_back_to_openalex(self):
        self.config.demo = False
        seen = []
        async def serve(url, **kw):
            seen.append((url, kw.get('headers', {})))
            if 'semanticscholar' in url:
                raise HTTPError('HTTP 429 from api.semanticscholar.org', True)
            return {'results': [{'id': 'https://openalex.org/W1', 'title': 'TOON study', 'doi': 'https://doi.org/10.1/x', 'publication_year': 2026,
                                 'abstract_inverted_index': {'tokens': [1], 'Fewer': [0]}, 'authorships': [{'author': {'display_name': 'A Author'}}],
                                 'best_oa_location': {'pdf_url': 'https://x.invalid/p.pdf'}}]}
        http = Transport(None)
        http.json = serve
        tools = Tools(self.config, self.store, self.journal, http)
        with patch.dict(os.environ):
            os.environ.pop('SEMANTIC_SCHOLAR_API_KEY', None)
            result = await tools.execute('a', 'search_papers', {'query': 'toon'})
        self.assertNotIn('x-api-key', seen[0][1])
        paper = result['results'][0]
        self.assertEqual(paper['text'], 'Fewer tokens')
        self.assertEqual((paper['provider'], paper['year'], paper['pdf_url']), ('openalex', 2026, 'https://x.invalid/p.pdf'))
        self.assertEqual(self.events('recovery')[0]['agent'], 'tools')

    async def test_fetching_a_known_url_is_progress_and_keeps_its_title(self):
        tools = Tools(self.config, self.store, self.journal, Transport(None))
        found = await tools.search_web('toon')
        self.assertEqual(found['results'][0]['title'], 'TOON page')
        self.assertEqual(found['results'][0]['text'], 'A snippet about TOON & JSON')
        first = await tools.fetch_page('https://page.invalid')
        self.assertTrue(first['progress'])
        source = self.store.state['sources'][first['source_id']]
        self.assertEqual((source['title'], source['evidence_type']), ('TOON page', 'page_text'))
        self.assertFalse((await tools.fetch_page('https://page.invalid'))['progress'])


class RestartTests(Fixture):
    @staticmethod
    def worker_prompt(body):
        return 'Your assignment:' in body['messages'][0]['content']

    async def test_replay_of_toon_run_replies_now_records_findings(self):
        """The two malformed reply shapes from the failed run now do real work."""
        sid = 'S' + hashlib.sha256(b'https://page.invalid').hexdigest()[:10]
        script = ['{"tool":"search_web","query":"TOON"}',
                  '```json\n[{"tool":"fetch_page","args":{"url":"https://page.invalid"}},'
                  f'{{"tool":"note","args":{{"text":"TOON reduces tokens per the page.","citations":["{sid}"]}}}}]\n```',
                  '{"tool":"finish","reason":"done"}']
        def model(body):
            return reply(script.pop(0) if self.worker_prompt(body) else '{"assignments":["Study TOON"]}')
        summary = await self.build(Transport(model)).run()
        self.assertEqual(summary['notes'], 1)
        self.assertEqual(summary['workers']['researcher-1'], 'done')
        self.assertEqual([r['method'] for r in summary['rounds']], ['agents'])
        self.assertEqual(self.events('error'), [])

    async def test_failed_worker_restarts_with_new_method_and_finds_evidence(self):
        sid, _ = self.store.source('Page', 'https://page.invalid', PAGE, 'brave', evidence_type='page_text')
        seen = []
        def model(body):
            if not self.worker_prompt(body):
                return reply('{"assignments":["Study TOON"]}')
            seen.append(body)
            if 'earlier attempt failed' not in body['messages'][0]['content']:
                return reply('this is not an action')
            return reply(json.dumps({'tool': 'note', 'args': {'text': 'A finding.', 'citations': [sid]}})) if len(self.store.state['notes']) == 0 else reply('{"tool":"finish"}')
        engine = self.build(Transport(model))
        summary = await engine.run()
        self.assertEqual([r['method'] for r in summary['rounds']], ['agents', 'agents-simplified'])
        self.assertEqual(summary['rounds'][1]['new_notes'], 1)
        self.assertEqual(summary['notes'], 1)
        restarted = seen[-1]
        self.assertEqual(restarted['chat_template_kwargs'], {'enable_thinking': False})
        self.assertIn('search_web', restarted['messages'][0]['content'])
        self.assertLess(engine.context_chars, self.config.context_chars)
        self.assertEqual(engine.pause.await_args_list[0].args, (5,))

    async def test_model_failures_are_not_fed_back_as_a_diagnosis_task(self):
        conversations = []
        def model(body):
            if self.worker_prompt(body):
                conversations.append(body['messages'])
            return reply('')
        self.config.max_restarts = 0
        await self.build(Transport(model)).run()
        self.assertTrue(conversations)
        self.assertFalse(any('failure' in m['content'] for c in conversations for m in c[2:]))

    async def test_dead_model_still_produces_labelled_evidence_and_paper(self):
        async def dead(url, **kw):
            if url.endswith('/chat/completions'):
                raise HTTPError('Connection failed to model', False)
            return await Transport(None).json(url, **kw)
        http = Transport(None)
        http.json = dead
        summary = await self.build(http).run()
        self.assertEqual([r['method'] for r in summary['rounds']], ['agents', 'agents-simplified', 'direct'])
        self.assertGreater(summary['notes'], 0)
        self.assertFalse(summary['model_synthesis'])
        self.assertGreater(summary['tool_completions']['search_web'], 0)
        self.assertTrue(all(n['agent'] == 'direct-excerpt' and n['text'].startswith('Verbatim excerpt') for n in self.store.state['notes']))
        self.assertTrue(all('footer' not in n['text'] for n in self.store.state['notes']))
        paper = (self.directory / 'paper.md').read_text()
        self.assertIn('methods used, in order: agents, agents-simplified, direct', paper)
        self.assertIn('not interpreted', paper)

    async def test_direct_round_uses_model_findings_when_it_can_answer(self):
        def model(body):
            system = body['messages'][0]['content']
            if 'Extract findings' in system:
                return reply('{"findings":["The source says TOON saves tokens."]}')
            if 'Write a rigorous' in system:
                sid = self.store.state['notes'][0]['citations'][0]
                return reply(f'# Paper\nThe source says TOON saves tokens [{sid}].')
            return reply('not json') if self.worker_prompt(body) else reply('{"assignments":["Study TOON"]}')
        summary = await self.build(Transport(model)).run()
        self.assertEqual(summary['rounds'][-1]['method'], 'direct')
        self.assertTrue(summary['model_synthesis'])
        notes = self.store.state['notes']
        self.assertTrue(notes)
        self.assertEqual({n['agent'] for n in notes}, {'direct'})
        self.assertEqual({n['text'] for n in notes}, {'The source says TOON saves tokens.'})

    async def test_direct_round_rereads_excerpt_only_sources_but_not_model_read_ones(self):
        excerpted, _ = self.store.source('Excerpted', 'https://a.invalid', PAGE, 'brave', evidence_type='page_text')
        read, _ = self.store.source('Read', 'https://b.invalid', PAGE, 'brave', evidence_type='page_text')
        self.store.note('Verbatim excerpt', [excerpted], 'direct-excerpt')
        self.store.note('Real finding', [read], 'direct')
        asked = []
        def model(body):
            asked.append(body['messages'][1]['content'])
            return reply('{"findings":["Now properly read."]}')
        await self.build(Transport(model)).direct_steps([])
        self.assertTrue(any('Source: Excerpted' in text for text in asked))
        self.assertFalse(any('Source: Read' in text for text in asked))

    async def test_writer_sees_every_note_while_workers_see_recent_ones(self):
        sid, _ = self.store.source('Page', 'https://page.invalid', PAGE, 'brave', evidence_type='page_text')
        for i in range(25):
            self.store.note(f'Finding number {i}', [sid], 'agent')
        self.assertEqual(len(self.store.evidence(40000)['notes']), 20)
        prompts = []
        def model(body):
            prompts.append(body['messages'][1]['content'])
            return reply(f'# Paper\nA finding [{sid}].')
        self.assertTrue(await self.build(Transport(model)).synthesize())
        self.assertIn('Finding number 0"', prompts[0])
        self.assertIn('Finding number 24"', prompts[0])

    async def test_no_restart_after_a_clean_finish_with_findings(self):
        sid, _ = self.store.source('Page', 'https://page.invalid', PAGE, 'brave', evidence_type='page_text')
        script = [json.dumps({'tool': 'note', 'args': {'text': 'Finding', 'citations': [sid]}}), '{"tool":"finish"}']
        def model(body):
            return reply(script.pop(0) if self.worker_prompt(body) else '{"assignments":["Study TOON"]}')
        engine = self.build(Transport(model))
        summary = await engine.run()
        self.assertEqual(len(summary['rounds']), 1)
        engine.pause.assert_not_awaited()

    async def test_restart_limit_is_respected(self):
        self.config.max_restarts = 1
        summary = await self.build(Transport(lambda body: reply('not json'))).run()
        self.assertEqual([r['method'] for r in summary['rounds']], ['agents', 'agents-simplified'])
        self.assertIn('Restart limit', self.events('recovery')[-1]['summary'])

    async def test_token_budget_ends_research_without_restarting(self):
        self.config.max_tokens = 3000
        def model(body):
            return {**reply('{"tool":"search_web","args":{"query":"q"}}'), 'usage': {'total_tokens': 1000}}
        engine = self.build(Transport(model))
        summary = await engine.run()
        self.assertTrue(engine.budget_exhausted)
        self.assertEqual(len(summary['rounds']), 1)
        self.assertEqual(summary['workers']['researcher-1'], 'LLM token budget reached')

    async def test_stop_during_recovery_pause_and_direct_round(self):
        async def slow_search(url, **kw):
            if url.endswith('/chat/completions'):
                return reply('not json')
            await asyncio.sleep(10)
        http = Transport(None)
        http.json = slow_search
        self.config.hours = 1
        engine = self.build(http)
        engine.config.max_restarts = 2
        task = asyncio.create_task(engine.run())
        for _ in range(500):
            await asyncio.sleep(.02)
            if self.events('round') and len(self.events('round')) == 3:
                break
        engine.request_stop()
        summary = await asyncio.wait_for(task, timeout=2)
        self.assertTrue(summary['stopped_by_user'])
        self.assertEqual(summary['rounds'][-1]['method'], 'direct')
        self.assertTrue((self.directory / 'paper.md').exists())

    async def test_deadline_is_never_exceeded_by_restarts(self):
        self.config.hours = .5 / 3600
        engine = self.build(Transport(lambda body: reply('not json')))
        del engine.pause
        started = time.monotonic()
        await engine.run()
        self.assertLess(time.monotonic() - started, .75)

    async def test_pause_ends_immediately_on_stop(self):
        self.config.hours = 1
        engine = self.build(Transport(None))
        del engine.pause
        waiting = asyncio.create_task(engine.pause(30))
        await asyncio.sleep(0)
        engine.stop.set()
        await asyncio.wait_for(waiting, timeout=.5)

    async def test_crashing_round_is_logged_and_restarted(self):
        engine = self.build(Transport(lambda body: reply('{"findings":["Found"]}')))
        async def crash(assignments):
            raise OSError('disk hiccup')
        engine.agent_round = crash
        await engine.research(['Study TOON'])
        self.assertIn('disk hiccup', self.events('error')[0]['summary'])
        self.assertEqual(engine.rounds[-1]['method'], 'direct')
        self.assertTrue(self.store.state['notes'])


class PlanRecoveryTests(Fixture):
    async def dead_plan(self):
        async def dead(url, **kw):
            raise HTTPError('Connection failed to model', False)
        http = Transport(None)
        http.json = dead
        return await self.build(http).plan()

    async def test_failed_planner_reuses_saved_plan(self):
        (self.directory / 'plan.json').write_text(json.dumps({'assignments': ['Measure TOON tokens']}))
        self.assertEqual(await self.dead_plan(), ['Measure TOON tokens'])
        self.assertIn('Using saved', self.events('recovery')[0]['summary'])

    async def test_unusable_saved_plan_falls_back_to_defaults(self):
        for saved in ('not json', '[]', '{"assignments":[]}', '{"assignments":[1]}', '{"assignments":["a","b"]}'):
            with self.subTest(saved=saved):
                (self.directory / 'plan.json').write_text(saved)
                self.assertIn('primary research papers', (await self.dead_plan())[0])

    async def test_direct_queries_stay_on_topic_and_never_repeat(self):
        engine = self.build(Transport(None))
        generic = 'Find primary research papers and define the central research questions.'
        queries = engine.direct_queries([generic])
        self.assertTrue(all('TOON' in query for query in queries), queries)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertEqual(engine.direct_queries([generic]), [])


class FakeResponse(io.BytesIO):
    def __init__(self, body, headers):
        super().__init__(body)
        self.headers = headers


class HttpWorkerTests(unittest.TestCase):
    """Run the helper in-process so these do not need a local socket."""
    def run_worker(self, opener):
        stdin = io.StringIO(json.dumps({'url': 'https://api.invalid/x', 'headers': {}, 'body': None, 'timeout': 1, 'max_bytes': 100}))
        stdout = io.StringIO()
        with patch('sys.stdin', stdin), patch('sys.stdout', stdout), patch('urllib.request.urlopen', opener):
            http_worker.main()
        return json.loads(stdout.getvalue())

    def test_short_body_is_a_retryable_error(self):
        result = self.run_worker(lambda *a, **k: FakeResponse(b'{"web":', {'Content-Length': '50'}))
        self.assertTrue(result['retryable'])
        self.assertIn('Truncated', result['error'])

    def test_complete_body_and_unsized_body_are_returned(self):
        for headers in ({'Content-Length': '7'}, {}):
            with self.subTest(headers=headers):
                result = self.run_worker(lambda *a, **k: FakeResponse(b'{"a":1}', headers))
                self.assertEqual(base64.b64decode(result['data']), b'{"a":1}')

    def test_oversized_body_is_still_rejected(self):
        result = self.run_worker(lambda *a, **k: FakeResponse(b'x' * 101, {'Content-Length': '101'}))
        self.assertFalse(result['retryable'])

    def test_chunked_connection_drop_is_a_retryable_error(self):
        def dropped(*a, **k):
            raise http.client.IncompleteRead(b'partial')
        self.assertTrue(self.run_worker(dropped)['retryable'])


def sse(*chunks, done=True):
    lines = [f'data: {json.dumps(chunk)}\n\n' for chunk in chunks] + (['data: [DONE]\n\n'] if done else [])
    return ''.join(lines).encode()


def delta(text=None, finish=None, reasoning=None):
    return {'choices': [{'delta': {'content': text, 'reasoning_content': reasoning}, 'finish_reason': finish}]}


class StreamingTests(unittest.TestCase):
    EVENT = {'Content-Type': 'text/event-stream'}

    def run_stream(self, opener):
        stdin = io.StringIO(json.dumps({'url': 'https://api.invalid/x', 'headers': {}, 'body': {'stream': True}, 'timeout': 30, 'max_bytes': 1000}))
        stdout = io.StringIO()
        with patch('sys.stdin', stdin), patch('sys.stdout', stdout), patch('urllib.request.urlopen', opener):
            http_worker.main()
        result = json.loads(stdout.getvalue())
        return json.loads(base64.b64decode(result['data'])) if 'data' in result else result

    def test_stream_collapses_to_a_normal_completion(self):
        body = sse(delta('Hel'), delta('lo', 'stop'), {'choices': [], 'usage': {'total_tokens': 42}})
        result = self.run_stream(lambda *a, **k: FakeResponse(body, self.EVENT))
        self.assertEqual(result['choices'][0]['message']['content'], 'Hello')
        self.assertEqual(result['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(result['usage']['total_tokens'], 42)

    def test_reasoning_only_stream_reports_empty_content(self):
        body = sse(delta(reasoning='thinking'), delta(finish='length'))
        message = self.run_stream(lambda *a, **k: FakeResponse(body, self.EVENT))['choices'][0]
        self.assertEqual(message['message']['content'], '')
        self.assertEqual(message['message']['reasoning_content'], 'thinking')
        self.assertEqual(message['finish_reason'], 'length')

    def test_stream_ending_early_keeps_text_and_is_marked_interrupted(self):
        body = sse(delta('Partial draft'), done=False)
        result = self.run_stream(lambda *a, **k: FakeResponse(body, self.EVENT))
        self.assertEqual(result['choices'][0]['message']['content'], 'Partial draft')
        self.assertEqual(result['choices'][0]['finish_reason'], 'interrupted')

    def test_connection_drop_keeps_text_but_fails_when_there_is_none(self):
        class Dropping(FakeResponse):
            def __iter__(self):
                yield from self.readlines()
                raise http.client.IncompleteRead(b'')
        result = self.run_stream(lambda *a, **k: Dropping(sse(delta('Partial'), done=False), self.EVENT))
        self.assertEqual(result['choices'][0]['finish_reason'], 'interrupted')
        result = self.run_stream(lambda *a, **k: Dropping(b'', self.EVENT))
        self.assertTrue(result['retryable'])

    def test_malformed_chunk_does_not_crash_the_helper(self):
        result = self.run_stream(lambda *a, **k: FakeResponse(b'data: {broken\n\n', self.EVENT))
        self.assertTrue(result['retryable'])

    def test_stream_size_limit_counts_text_not_protocol_overhead(self):
        small = sse(*[delta('x') for _ in range(50)], delta(finish='stop'))
        self.assertGreater(len(small), 1000)
        self.assertEqual(len(self.run_stream(lambda *a, **k: FakeResponse(small, self.EVENT))['choices'][0]['message']['content']), 50)
        result = self.run_stream(lambda *a, **k: FakeResponse(sse(delta('x' * 1001)), self.EVENT))
        self.assertFalse(result['retryable'])

    def test_server_ignoring_stream_request_still_works(self):
        result = self.run_stream(lambda *a, **k: FakeResponse(b'{"choices":[]}', {'Content-Type': 'application/json'}))
        self.assertEqual(result, {'choices': []})


class StreamingClientTests(Fixture):
    async def chat(self, serve, final=False):
        http = Transport(None)
        http.json = serve
        self.llm = LLM(self.config, self.journal, http)
        return await self.llm.chat('a', [{'role': 'user', 'content': 'q'}], final=final)

    async def test_requests_stream_and_ask_for_usage(self):
        bodies = []
        async def serve(url, **kw):
            bodies.append(kw['body'])
            return reply('answer')
        await self.chat(serve)
        self.assertIs(bodies[0]['stream'], True)
        self.assertEqual(bodies[0]['stream_options'], {'include_usage': True})

    async def test_server_rejecting_streaming_falls_back(self):
        bodies = []
        async def serve(url, **kw):
            bodies.append(kw['body'])
            if kw['body']['stream']:
                raise HTTPError('HTTP 400 from model', False)
            return reply('answer')
        self.assertEqual(await self.chat(serve), 'answer')
        self.assertNotIn('stream_options', bodies[-1])
        self.assertFalse(self.llm.streaming)

    async def test_null_usage_is_estimated(self):
        async def serve(url, **kw):
            return {**reply('answer'), 'usage': None}
        await self.chat(serve)
        self.assertGreater(self.llm.tokens, 0)

    async def test_interrupted_decision_is_retried(self):
        replies = [reply('{"tool":"sea', 'interrupted'), reply('{"tool":"finish"}')]
        async def serve(url, **kw):
            return replies.pop(0)
        self.assertEqual(await self.chat(serve), '{"tool":"finish"}')

    async def test_interrupted_paper_keeps_longest_draft_and_is_labelled(self):
        replies = [reply('# Long partial draft', 'interrupted'), reply('# Short', 'interrupted'), reply('', 'interrupted')]
        async def serve(url, **kw):
            return replies.pop(0)
        answer = await self.chat(serve, final=True)
        self.assertIn('# Long partial draft', answer)
        self.assertIn('cut off', answer)

    async def test_failure_after_partial_paper_still_returns_it(self):
        calls = []
        async def serve(url, **kw):
            calls.append(1)
            if len(calls) == 1:
                return reply('# Partial', 'interrupted')
            raise HTTPError('Connection failed to model', False)
        answer = await self.chat(serve, final=True)
        self.assertIn('# Partial', answer)
        self.assertIn('cut off', answer)

    async def test_complete_paper_is_not_labelled(self):
        async def serve(url, **kw):
            return reply('# Whole paper')
        self.assertEqual(await self.chat(serve, final=True), '# Whole paper')


class NoteEnforcementTests(Fixture):
    @staticmethod
    def worker_prompt(body):
        return 'Your assignment:' in body['messages'][0]['content']

    def wandering_model(self, seen):
        counter = iter(range(1000))
        def model(body):
            if not self.worker_prompt(body):
                return reply('{"assignments":["Study TOON"]}' if 'plan literature' in body['messages'][0]['content'] else '{"findings":["The source reports fewer tokens."]}')
            seen.append(body['messages'])
            return reply(json.dumps({'tool': 'read_file', 'args': {'path': f'missing-{next(counter)}'}}))
        return model

    async def test_worker_is_told_to_save_findings_then_stopped_and_restarted(self):
        seen = []
        tools_ok = {'text': 'x', 'progress': True}
        engine = self.build(Transport(self.wandering_model(seen)))
        engine.tools.read_file = AsyncMock(return_value=tools_ok)
        summary = await engine.run()
        self.assertEqual([r['method'] for r in summary['rounds']], ['agents', 'agents-simplified', 'direct'])
        self.assertTrue(any('must be note' in m['content'] for c in seen for m in c))
        self.assertLessEqual(engine.tools.read_file.await_count, 20, 'each agent round stops after ten actions')
        self.assertTrue(self.store.state['notes'])
        self.assertTrue(all(n['agent'] == 'direct' for n in self.store.state['notes']))

    async def test_saving_a_finding_resets_the_nudge(self):
        sid, _ = self.store.source('Page', 'https://page.invalid', PAGE, 'brave', evidence_type='page_text')
        actions = iter([{'tool': 'read_file', 'args': {'path': 'a'}}, {'tool': 'read_file', 'args': {'path': 'b'}}, {'tool': 'read_file', 'args': {'path': 'c'}},
                        {'tool': 'note', 'args': {'text': 'Finding', 'citations': [sid]}}, {'tool': 'read_file', 'args': {'path': 'd'}}, {'tool': 'finish', 'args': {}}])
        seen = []
        def model(body):
            if not self.worker_prompt(body):
                return reply('{"assignments":["Study TOON"]}')
            seen.append(body['messages'])
            return reply(json.dumps(next(actions)))
        engine = self.build(Transport(model))
        engine.tools.read_file = AsyncMock(return_value={'text': 'x', 'progress': True})
        summary = await engine.run()
        self.assertEqual(len(summary['rounds']), 1)
        self.assertIn('must be note', seen[3][-1]['content'])
        self.assertNotIn('must be note', seen[5][-1]['content'])

    async def test_excerpts_alone_do_not_end_recovery_but_do_after_direct(self):
        sid, _ = self.store.source('Page', 'https://page.invalid', PAGE, 'brave', evidence_type='page_text')
        self.store.note('Verbatim excerpt', [sid], 'direct-excerpt')
        engine = self.build(Transport(None))
        self.assertIn('model-read', engine.restart_reason('agents'))
        self.assertEqual(engine.restart_reason('direct'), '')
        self.store.note('Model finding', [sid], 'direct')
        self.assertEqual(engine.restart_reason('agents'), '')

    async def test_model_reading_replaces_excerpts_and_papers_go_first(self):
        snippet, _ = self.store.source('Snippet', 'https://a.invalid', PAGE, 'brave', evidence_type='search_snippet')
        paper, _ = self.store.source('Paper', 'https://b.invalid', PAGE, 'openalex', evidence_type='abstract')
        self.store.note('Verbatim excerpt about paper', [paper], 'direct-excerpt')
        self.store.note('Verbatim excerpt about snippet', [snippet], 'direct-excerpt')
        asked = []
        def model(body):
            asked.append(body['messages'][1]['content'])
            return reply('{"findings":[]}' if 'Source: Paper' in asked[-1] else '{"findings":["Read properly."]}')
        await self.build(Transport(model)).direct_steps([])
        self.assertIn('Source: Paper', asked[0])
        self.assertEqual([n['text'] for n in self.store.state['notes']], ['Read properly.'])

    async def test_clean_finish_with_unread_sources_goes_straight_to_direct(self):
        read, _ = self.store.source('Read', 'https://a.invalid', PAGE, 'brave', evidence_type='page_text')
        unread, _ = self.store.source('Unread case study', 'https://b.invalid', PAGE, 'brave', evidence_type='page_text')
        script = [json.dumps({'tool': 'note', 'args': {'text': 'Finding', 'citations': [read]}}), '{"tool":"finish"}']
        asked = []
        def model(body):
            if self.worker_prompt(body):
                return reply(script.pop(0))
            if 'Extract findings' in body['messages'][0]['content']:
                asked.append(body['messages'][1]['content'])
                return reply('{"findings":["Case study result."]}')
            return reply('{"assignments":["Study TOON"]}')
        summary = await self.build(Transport(model)).run()
        self.assertEqual([r['method'] for r in summary['rounds']], ['agents', 'direct'])
        self.assertTrue(any('Source: Unread case study' in text for text in asked))
        self.assertFalse(any('Source: Read\n' in text for text in asked))
        self.assertIn('Case study result.', [n['text'] for n in self.store.state['notes']])

    async def test_irrelevant_source_is_not_reread_on_resume(self):
        sid, _ = self.store.source('Off topic', 'https://a.invalid', PAGE, 'brave', evidence_type='page_text')
        self.store.source('Other', 'https://b.invalid', PAGE, 'brave', evidence_type='page_text')
        engine = self.build(Transport(lambda body: reply('{"findings":[]}')))
        await engine.extract_findings(self.store.state['sources'][sid])
        self.assertNotIn(sid, [s['id'] for s in engine.unread_sources()])
        restored = Store(self.directory, TOPIC, self.journal)
        self.assertIn(sid, restored.state['read_sources'])

    async def test_unproductive_stops_count_as_failures_and_restart(self):
        sid, _ = self.store.source('Page', 'https://page.invalid', PAGE, 'brave', evidence_type='page_text')
        self.store.note('Earlier finding', [sid], 'researcher-1')
        def model(body):
            if not self.worker_prompt(body):
                return reply('{"assignments":["Study TOON"]}')
            if 'earlier attempt failed' in body['messages'][0]['content']:
                return reply('{"tool":"finish"}')
            return reply('{"tool":"read_file","args":{"path":"same"}}')
        engine = self.build(Transport(model))
        engine.tools.read_file = AsyncMock(return_value={'text': 'x', 'progress': False})
        summary = await engine.run()
        self.assertEqual([r['method'] for r in summary['rounds']], ['agents', 'agents-simplified'])

    async def test_failed_model_keeps_existing_excerpts(self):
        sid, _ = self.store.source('Page', 'https://a.invalid', PAGE, 'brave', evidence_type='page_text')
        self.store.note('Old excerpt', [sid], 'direct-excerpt')
        await self.build(Transport(lambda body: reply('not json'))).extract_findings(self.store.state['sources'][sid])
        self.assertIn('Old excerpt', [n['text'] for n in self.store.state['notes']])


class InvalidJsonTests(unittest.IsolatedAsyncioTestCase):
    async def test_bad_json_body_is_retryable(self):
        client = HTTP(1)
        client.request = AsyncMock(return_value=b'{"web": {"title": "cut off')
        with self.assertRaises(HTTPError) as caught:
            await client.json('https://api.invalid/x')
        self.assertTrue(caught.exception.retryable)
        self.assertIn('api.invalid', str(caught.exception))


class RedactionTests(unittest.TestCase):
    def test_short_placeholder_keys_do_not_corrupt_evidence(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'LOCAL_LLM_API_KEY': 'none', 'BRAVE_SEARCH_API_KEY': 'a-real-long-secret'}, clear=True):
            journal = Journal(temp, quiet=True)
            self.assertEqual(journal.clean('none of these a-real-long-secret'), 'none of these [REDACTED]')


class HelperTests(unittest.TestCase):
    def test_key_terms_keep_content_words_once(self):
        terms = key_terms('How can TOON (Token-Oriented Object Notation) improve TOON compared to JSON?')
        self.assertEqual(terms, ['TOON', 'Token-Oriented', 'Object', 'Notation', 'improve', 'JSON'])
        self.assertEqual(len(key_terms('alpha beta gamma delta', limit=2)), 2)

    def test_excerpt_rejects_generic_and_substring_matches(self):
        terms = key_terms('How can TOON (Token-Oriented Object Notation) improve token efficiency and reduce processing costs compared to JSON?')
        platooning = 'The scope of platooning is to improve safety, reduce fuel consumption, and increase road use efficiency.'
        self.assertEqual(excerpt_note(platooning, terms), '')
        self.assertIn('fewer tokens', excerpt_note('TOON matches retrieval accuracy while using 42.6% fewer tokens than before.', terms))

    def test_excerpt_without_named_subject_uses_all_terms(self):
        terms = key_terms('why does ice float in liquid water')
        self.assertIn('less dense', excerpt_note('Solid ice is less dense than liquid water at normal pressure.', terms))
        self.assertEqual(excerpt_note('A completely unrelated sentence about ice cream flavors here.', terms), '')

    def test_excerpt_requires_relevant_sentences(self):
        self.assertEqual(excerpt_note('Cookies are used on this website to improve your experience.', ['toon', 'json']), '')
        quote = excerpt_note(PAGE, ['TOON', 'JSON', 'token'])
        self.assertIn('TOON improves token efficiency', quote)
        self.assertNotIn('footer', quote)


if __name__ == '__main__':
    unittest.main()

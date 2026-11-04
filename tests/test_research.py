import asyncio
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, AsyncMock

from agen1.config import Config
from agen1.demo import DemoHTTP
from agen1.engine import Engine
from agen1.network import LLM, HTTPError, parse_action
from agen1.storage import Journal, Store
from agen1.tools import Tools


class ResearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        # These tests pin single-round behavior; test_recovery.py covers restarts.
        self.config = Config('Evidence synthesis', self.directory, hours=.01, demo=True,
                             supervisor_interval=.02, request_timeout=1, max_restarts=0)
        self.journal = Journal(self.directory, quiet=True)
        self.store = Store(self.directory, self.config.topic, self.journal)
        self.http = DemoHTTP()
        self.tools = Tools(self.config, self.store, self.journal, self.http)
        self.llm = LLM(self.config, self.journal, self.http)

    def events(self):
        return [json.loads(line) for line in self.journal.path.read_text().splitlines()]

    async def test_parallel_end_to_end_and_resume(self):
        result = await Engine(self.config, self.llm, self.tools, self.store, self.journal).run()
        self.assertEqual(result['notes'], 3)
        self.assertEqual(result['sources'], 1)
        self.assertTrue(result['model_synthesis'])
        paper = (self.directory / 'paper.md').read_text()
        self.assertIn('References', paper)
        self.assertIn('synthetic', paper.lower())
        events = self.events()
        worker_calls = [e for e in events if e['kind'] == 'tool_call']
        self.assertEqual(len(worker_calls), 9)
        first_result = next(i for i,e in enumerate(events) if e['kind'] == 'tool_result')
        self.assertGreaterEqual(len({e['agent'] for e in events[:first_result] if e['kind'] == 'tool_call'}), 2)
        restored = Store(self.directory, self.config.topic, self.journal)
        self.assertEqual(len(restored.state['notes']), 3)
        self.assertEqual(restored.state['sessions'], 2)
        self.assertIn('Calling search_papers', self.journal.activity.read_text())

    async def test_identical_call_loop_stopped(self):
        original = self.http.json
        async def repeat(url, **kw):
            if url.endswith('/chat/completions') and 'Your assignment:' in kw['body']['messages'][0]['content']:
                return {'choices': [{'message': {'content': '{"tool":"read_file","args":{"path":"missing"}}'}}]}
            return await original(url, **kw)
        self.http.json = repeat
        self.config.workers = 1
        result = await Engine(self.config, self.llm, self.tools, self.store, self.journal).run()
        self.assertIn('Repeated identical', result['workers']['researcher-1'])
        self.assertTrue((self.directory / 'paper.md').exists())
        self.assertEqual(len([e for e in self.events() if e['kind'] == 'tool_call']), 2)

    async def test_tool_error_is_returned_to_model_and_repaired(self):
        original = self.http.json
        observed = []
        async def recover(url, **kw):
            messages = kw.get('body', {}).get('messages', [])
            if messages and 'Your assignment:' in messages[0]['content']:
                observed.append(messages)
                turns = sum(m['role'] == 'assistant' for m in messages)
                action = {'tool': 'does_not_exist', 'args': {}} if turns == 0 else {'tool': 'finish', 'args': {'reason': 'Recovered from bad tool'}}
                return {'choices': [{'message': {'content': json.dumps(action)}}]}
            return await original(url, **kw)
        self.http.json = recover
        self.config.workers = 1
        result = await Engine(self.config, self.llm, self.tools, self.store, self.journal).run()
        self.assertIn('Recovered', result['workers']['researcher-1'])
        self.assertIn('Unknown tool', observed[1][-1]['content'])

    async def test_deadline_cancels_slow_work_and_writes_fallback(self):
        self.config.hours = .0001
        async def slow(*args, **kwargs):
            await asyncio.sleep(10)
        self.http.json = slow
        start = time.monotonic()
        result = await Engine(self.config, self.llm, self.tools, self.store, self.journal).run()
        self.assertLess(time.monotonic() - start, 1)
        self.assertFalse(result['model_synthesis'])
        self.assertIn('No validated', (self.directory / 'paper.md').read_text())

    async def test_watchdog_cancels_stalled_agent(self):
        self.config.workers = 1
        self.config.stall_seconds = .05
        original = self.http.json
        async def slow_worker(url, **kw):
            if 'Your assignment:' in kw['body']['messages'][0]['content']:
                await asyncio.sleep(10)
            return await original(url, **kw)
        self.http.json = slow_worker
        result = await Engine(self.config, self.llm, self.tools, self.store, self.journal).run()
        self.assertIn('stall window', result['workers']['researcher-1'])
        self.assertTrue(any(e['agent'] == 'supervisor' and e['kind'] == 'intervention' for e in self.events()))

    async def test_paths_shell_permissions_and_subprocess(self):
        with self.assertRaises(PermissionError):
            await self.tools.write_file('../outside.txt', 'bad')
        link = self.tools.workspace / 'escape'
        link.symlink_to(self.directory)
        with self.assertRaises(PermissionError):
            await self.tools.write_file('escape/outside.txt', 'bad')
        with self.assertRaises(PermissionError):
            await self.tools.shell('true')
        self.config.allow_shell = True
        result = await self.tools.shell("printf 'ok'")
        self.assertEqual(result['output'], 'ok')
        self.assertEqual(result['exit_code'], 0)
        self.config.request_timeout = .05
        with self.assertRaises(TimeoutError):
            await self.tools.shell('sleep 10')

    async def test_redaction_notes_citations_and_resume_topic(self):
        with patch.dict(os.environ, {'BRAVE_SEARCH_API_KEY': 'test-secret-value'}):
            journal = Journal(self.directory, quiet=True)
            journal.event('agent', 'tool', 'test-secret-value', output='Bearer private-token')
        text = journal.path.read_text() + journal.activity.read_text()
        self.assertNotIn('test-secret-value', text)
        self.assertNotIn('private-token', text)
        with self.assertRaises(ValueError):
            self.store.note('A claim', ['Sunknown'], 'agent')
        with self.assertRaises(ValueError):
            Store(self.directory, 'different topic', self.journal)

    async def test_hallucinated_source_uses_fallback(self):
        sid, _ = self.store.source('real fixture', 'https://example.invalid', 'Evidence', 'fixture')
        self.store.note('Evidence-backed note', [sid], 'agent')
        async def bad(*args, **kwargs):
            return {'choices': [{'message': {'content': '# Paper\nUnsupported claim [Sfake]'}}]}
        self.http.json = bad
        generated = await Engine(self.config, self.llm, self.tools, self.store, self.journal).synthesize()
        self.assertFalse(generated)
        self.assertNotIn('[Sfake]', (self.directory / 'paper.md').read_text())

    async def test_shared_token_budget(self):
        self.config.max_tokens = 10
        with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
            await self.llm.chat('test', [{'role': 'user', 'content': 'hello'}])
        self.assertEqual(self.llm.reserved, 0)

    async def test_provider_headers_and_evidence_types(self):
        self.config.demo = False
        captured = []
        async def fake(url, **kw):
            captured.append((url, kw))
            if 'semanticscholar' in url:
                return {'data': [{'paperId': '123', 'title': 'Paper', 'abstract': 'Abstract', 'url': 'https://paper.invalid', 'authors': [{'name': 'A'}], 'openAccessPdf': {'url': 'https://paper.invalid/p.pdf'}}]}
            return {'web': {'results': [{'title': 'Page', 'url': 'https://page.invalid', 'description': 'Snippet'}]}}
        self.http.json = fake
        with patch.dict(os.environ, {'SEMANTIC_SCHOLAR_API_KEY': 'ss-test', 'BRAVE_SEARCH_API_KEY': 'brave-test'}):
            papers = await self.tools.search_papers('test query')
            pages = await self.tools.search_web('test query')
        self.assertEqual(captured[0][1]['headers']['x-api-key'], 'ss-test')
        self.assertEqual(captured[1][1]['headers']['X-Subscription-Token'], 'brave-test')
        self.assertIn('query=test+query', captured[0][0])
        self.assertEqual(papers['results'][0]['evidence_type'], 'abstract')
        self.assertEqual(pages['results'][0]['evidence_type'], 'search_snippet')

    async def test_retryable_http_error_recovers(self):
        calls = []
        async def transient(*args, **kw):
            calls.append(1)
            if len(calls) == 1:
                raise HTTPError('HTTP 429', True)
            return {'choices': [{'message': {'content': 'recovered'}}]}
        self.http.json = transient
        answer = await self.llm.chat('test', [{'role': 'user', 'content': 'hello'}])
        self.assertEqual(answer, 'recovered')
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.llm.reserved, 0)

    async def test_stop_interrupts_planning_and_saves_report(self):
        async def slow(*args, **kwargs):
            await asyncio.sleep(10)
        self.http.json = slow
        engine = Engine(self.config, self.llm, self.tools, self.store, self.journal)
        task = asyncio.create_task(engine.run())
        await asyncio.sleep(.03)
        engine.request_stop()
        result = await asyncio.wait_for(task, timeout=.5)
        self.assertTrue(result['stopped_by_user'])
        self.assertTrue((self.directory / 'paper.md').exists())

    async def test_supervisor_model_redirects_and_stops(self):
        from agen1.engine import Worker
        engine = Engine(self.config, self.llm, self.tools, self.store, self.journal)
        engine.workers = {'researcher-1': Worker('researcher-1', 'Original assignment'),
                          'researcher-2': Worker('researcher-2', 'Another assignment')}
        sleeping = asyncio.create_task(asyncio.sleep(10))
        engine.tasks['researcher-2'] = sleeping
        async def review(*args, **kwargs):
            engine.stop.set()
            return json.dumps({'interventions': [
                {'agent': 'researcher-1', 'action': 'redirect', 'assignment': 'Compare contradictory evidence', 'reason': 'Coverage gap'},
                {'agent': 'researcher-2', 'action': 'stop', 'reason': 'Repeated unproductive searches'}]})
        engine.bounded_chat = review
        with patch('agen1.engine.asyncio.sleep', new=AsyncMock()):
            await engine.review_work()
        await asyncio.gather(sleeping, return_exceptions=True)
        self.assertEqual(engine.workers['researcher-1'].task, 'Compare contradictory evidence')
        self.assertEqual(engine.workers['researcher-1'].redirects, 1)
        self.assertTrue(sleeping.cancelled())

    async def test_pdf_download_and_evidence_not_downgraded(self):
        sid, _ = self.store.source('Paper', 'https://paper.invalid', 'Abstract', 'semantic_scholar',
            paper_id='pdf1', evidence_type='abstract', pdf_url='https://paper.invalid/file.pdf')
        async def pdf(*args, **kwargs):
            return b'%PDF-1.4\nminimal fixture'
        self.http.request = pdf
        with patch('agen1.tools.importlib.util.find_spec', return_value=None):
            result = await self.tools.download_paper(sid)
        self.assertFalse(result['extracted'])
        self.assertTrue(Path(result['path']).read_bytes().startswith(b'%PDF'))
        self.store.state['sources'][sid].update(text='Extracted full text', evidence_type='pdf_excerpt')
        self.store.source('Paper', 'https://paper.invalid', 'Abstract again', 'semantic_scholar', paper_id='pdf1', evidence_type='abstract')
        self.assertEqual(self.store.state['sources'][sid]['text'], 'Extracted full text')
        async def not_pdf(*args, **kwargs):
            return b'<html>Access denied</html>'
        self.http.request = not_pdf
        self.store.state['sources'][sid]['pdf_url'] = 'https://paper.invalid/file.pdf'
        with self.assertRaisesRegex(ValueError, 'not a PDF'):
            await self.tools.download_paper(sid)

    async def test_optional_real_pdf_text_extraction(self):
        try:
            from pypdf import PdfWriter
            from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject
        except ImportError:
            self.skipTest('Install the pdf extra to exercise real PDF extraction')
        import io
        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 50 700 Td (Verified extraction fixture) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        buffer = io.BytesIO()
        writer.write(buffer)
        async def pdf(*args, **kwargs):
            return buffer.getvalue()
        self.http.request = pdf
        sid, _ = self.store.source('PDF fixture', 'https://pdf.invalid', 'abstract', 'fixture', pdf_url='https://pdf.invalid/file.pdf')
        self.config.request_timeout = 3
        result = await self.tools.download_paper(sid)
        self.assertIn('Verified extraction fixture', result['text'])
        self.assertTrue(result['progress'])
        self.assertEqual(self.store.state['sources'][sid]['evidence_type'], 'pdf_excerpt')

    def test_config_and_json_action_validation(self):
        self.config.hours = float('nan')
        with self.assertRaises(ValueError):
            self.config.validate()
        self.assertEqual(parse_action('```json\n{"tool":"finish","args":{}}\n```')['tool'], 'finish')
        with self.assertRaises(ValueError):
            parse_action('{"tool":1}')


if __name__ == '__main__':
    unittest.main()

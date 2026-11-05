"""Short-budget regression tests use deterministic models, never live research."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest

from agen1.config import Config
from agen1.demo import DemoHTTP
from agen1.engine import Engine, Worker
from agen1.network import LLM
from agen1.storage import Journal, Store
from agen1.tools import Tools


QUESTION = 'Why does ice float in liquid water?'


class ScenarioLLM:
    tokens = 0
    context_limit = 100_000
    peak_prompt_tokens = 0
    trims = 0

    def prompt_chars(self, final=False):
        return 300_000

    def __init__(self, source_id, worker_delay=0, writer_delay=0):
        self.source_id = source_id
        self.worker_delay = worker_delay
        self.writer_delay = writer_delay
        self.requests = []
        self.cancelled = []
        self.turns = 0

    async def chat(self, agent, messages, final=False):
        self.requests.append(agent)
        try:
            if agent == 'planner':
                return json.dumps({'assignments': ['Compare solid and liquid water density.']})
            if agent == 'writer':
                await asyncio.sleep(self.writer_delay)
                return f'# Fixture paper\nSynthetic source says ice has lower density [{self.source_id}].'
            self.turns += 1
            if self.turns == 1:
                return json.dumps({'tool': 'note', 'args': {'text': 'Fixture says ice is less dense than liquid water.', 'citations': [self.source_id]}})
            await asyncio.sleep(self.worker_delay)
            return json.dumps({'tool': 'finish', 'args': {'reason': 'Fixture assignment completed'}})
        except asyncio.CancelledError:
            self.cancelled.append(agent)
            raise


class DeadlineTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, seconds):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        directory = Path(temp.name)
        config = Config(QUESTION, directory, hours=seconds / 3600, workers=1,
                        supervisor_interval=.02, request_timeout=1, demo=True)
        journal = Journal(directory, quiet=True)
        store = Store(directory, QUESTION, journal)
        sid, _ = store.source('Synthetic water fixture', 'https://example.invalid/water',
                              'Ice is less dense than liquid water.', 'fixture')
        tools = Tools(config, store, journal, DemoHTTP())
        return config, journal, store, tools, sid

    async def test_research_deadlines_reserve_writing_at_multiple_budgets(self):
        for seconds in (.2, .5, 1):
            with self.subTest(seconds=seconds):
                config, journal, store, tools, sid = self.fixture(seconds)
                llm = ScenarioLLM(sid, worker_delay=10)
                engine = Engine(config, llm, tools, store, journal)
                started = time.monotonic()
                summary = await engine.run()
                elapsed = time.monotonic() - started
                self.assertGreaterEqual(elapsed, seconds * .7)
                self.assertLess(elapsed, seconds + .25)
                self.assertTrue(summary['model_synthesis'])
                self.assertEqual(summary['notes'], 1)
                self.assertIn('researcher-1', llm.cancelled)
                self.assertIn('time budget', summary['workers']['researcher-1'])
                self.assertTrue(all(task.done() for task in engine.tasks.values()))

    async def test_writer_timeout_produces_fallback_at_multiple_budgets(self):
        for seconds in (.2, .5, 1):
            with self.subTest(seconds=seconds):
                config, journal, store, tools, sid = self.fixture(seconds)
                llm = ScenarioLLM(sid, writer_delay=10)
                engine = Engine(config, llm, tools, store, journal)
                started = time.monotonic()
                summary = await engine.run()
                elapsed = time.monotonic() - started
                self.assertGreaterEqual(elapsed, seconds * .9)
                self.assertLess(elapsed, seconds + .25)
                self.assertFalse(summary['model_synthesis'])
                self.assertIn('writer', llm.cancelled)
                paper = (config.run_dir / 'paper.md').read_text()
                self.assertIn('attempted 0 search tool calls', paper)
                self.assertIn(sid, paper)
                self.assertNotIn('Parallel agents searched', paper)

    async def test_expired_budget_never_starts_model_or_tool(self):
        config, journal, store, tools, sid = self.fixture(1)
        llm = ScenarioLLM(sid)
        engine = Engine(config, llm, tools, store, journal)
        engine.research_deadline = time.monotonic() - 1
        with self.assertRaises(TimeoutError):
            await engine.bounded_chat('researcher-1', [])
        self.assertEqual(llm.requests, [])
        engine.research_deadline = time.monotonic() + 1
        async def deadline_response(*args, **kwargs):
            engine.research_deadline = time.monotonic() - 1
            return json.dumps({'tool': 'note', 'args': {'text': 'Too late', 'citations': [sid]}})
        engine.bounded_chat = deadline_response
        state = Worker('researcher-1', 'Study density')
        await engine.worker(state)
        self.assertEqual(store.state['notes'], [])
        self.assertEqual(dict(engine.tool_attempts), {})
        self.assertEqual(state.stopped, 'Research time budget reached')

    async def test_no_work_fallback_distinguishes_supplied_material(self):
        config, journal, store, tools, sid = self.fixture(1)
        store.source('Proposal', 'materials/proposal.txt', 'Investigate water density.', 'user', evidence_type='user_proposal')
        engine = Engine(config, ScenarioLLM(sid), tools, store, journal)
        paper = engine.fallback_paper()
        self.assertIn('attempted 0 search tool calls', paper)
        self.assertIn('including 1 user-supplied materials', paper)
        self.assertIn('not been independently verified', paper)
        self.assertNotIn('Parallel agents searched', paper)

    async def test_stop_during_tool_cancels_and_saves(self):
        config, journal, store, tools, sid = self.fixture(30)
        cancelled = []
        async def slow_tool(*args, **kwargs):
            try:
                await asyncio.sleep(20)
            finally:
                cancelled.append(True)
        tools.execute = slow_tool
        engine = Engine(config, ScenarioLLM(sid), tools, store, journal)
        task = asyncio.create_task(engine.run())
        await asyncio.sleep(.05)
        engine.request_stop()
        summary = await asyncio.wait_for(task, timeout=.5)
        self.assertTrue(cancelled)
        self.assertTrue(summary['stopped_by_user'])
        self.assertTrue((config.run_dir / 'paper.md').exists())
        self.assertTrue(all(task.done() for task in engine.tasks.values()))


async def demo_matrix(output_directory):
    """Run the real scheduling/tools pipeline with synthetic provider responses."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for seconds in (1, 3, 6):
        directory = output_directory / f'demo-{seconds}s'
        config = Config(QUESTION, directory, hours=seconds / 3600, demo=True)
        journal = Journal(directory, quiet=True)
        store = Store(directory, QUESTION, journal)
        http = DemoHTTP()
        llm = LLM(config, journal, http)
        tools = Tools(config, store, journal, http)
        engine = Engine(config, llm, tools, store, journal)
        summary = await engine.run()
        assert summary['elapsed_seconds'] < seconds + .5, summary
        assert (directory / 'paper.md').exists()
        assert (directory / 'activity.md').exists()
        assert all(task.done() for task in engine.tasks.values())
        rows.append({'budget_seconds': seconds, **summary})
    (output_directory / 'matrix.json').write_text(json.dumps(rows, indent=2))
    lines = ['# Short-budget research validation', '', f'Question: **{QUESTION}**', '',
             'All models and provider responses in these runs are deterministic offline fixtures. These runs validate scheduling and artifacts, not scientific results or live integrations.', '',
             '| Budget | Elapsed | Sources | Notes | Model-written draft |', '| --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['budget_seconds']} s | {row['elapsed_seconds']:.3f} s | {row['sources']} | {row['notes']} | {row['model_synthesis']} |")
    lines += ['', 'Additional regression tests force research and writer timeouts at 0.2, 0.5, and 1 second, check cancellation and saved fallback reports, reject new work after expiry, and stop during an active tool call.']
    (output_directory / 'README.md').write_text('\n'.join(lines) + '\n')
    return rows


if __name__ == '__main__':
    unittest.main()

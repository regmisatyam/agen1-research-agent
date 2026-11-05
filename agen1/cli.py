import argparse
from contextlib import contextmanager
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys

from .config import Config
from .materials import import_materials
from .demo import DemoHTTP
from .engine import Engine
from .network import HTTP, LLM
from .storage import Journal, Store
from .tools import Tools


def parser():
    p = argparse.ArgumentParser(description='Autonomous, local-first multi-agent research')
    commands = p.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Start or resume a research run')
    run.add_argument('topic', nargs='?', help='Research question; optional when resuming')
    duration = run.add_mutually_exclusive_group()
    duration.add_argument('--hours', type=float, help='Maximum research duration, including writing (default: 1 hour)')
    duration.add_argument('--seconds', type=float, help='Set a short duration in seconds, including writing')
    run.add_argument('--proposal', type=Path, action='append', default=[], help='Research proposal file; repeat for multiple proposals')
    run.add_argument('--material', type=Path, action='append', default=[], help='Supporting material file; repeat for multiple files')
    run.add_argument('--run-dir', type=Path, help='Persistent output directory; existing evidence is resumed')
    run.add_argument(
    '--base-url',
    default=os.getenv('AGEN1_BASE_URL', 'http://127.0.0.1:11434/v1'),
    help='LLM API base URL (defaults to AGEN1_BASE_URL environment variable)',
    )

    run.add_argument(
    '--model',
    default=os.getenv('AGEN1_MODEL', 'qwen2.5:14b'),
    help='LLM model (defaults to AGEN1_MODEL environment variable)'
    )
    run.add_argument('--workers', type=int, default=3)
    run.add_argument('--llm-concurrency', type=int, default=2)
    run.add_argument('--request-timeout', type=float, default=60)
    run.add_argument('--max-steps', type=int, default=500, help='Per-worker action limit')
    run.add_argument('--max-tokens', type=int, default=0, help='Optional cap on LLM tokens across the whole session; 0 (default) leaves the time budget in charge')
    run.add_argument('--context-limit', type=int, default=100000, help="The model's token window for one request, prompt and output together; requests are compacted to fit")
    run.add_argument('--max-output-tokens', type=int, default=2048)
    run.add_argument('--context-chars', type=int, default=40000, help='Soft character budget for the evidence and history in each researcher request')
    run.add_argument('--stall-seconds', type=float, default=180)
    run.add_argument('--max-restarts', type=int, default=5, help='Recovery rounds, each using a different method, after workers fail or no findings are recorded')
    run.add_argument('--no-reasoning', action='store_true', help='Ask the server to skip model reasoning from the start (chat_template_kwargs enable_thinking=false)')
    run.add_argument('--allow-shell', action='store_true', help='Allow autonomous shell commands with your OS permissions (not sandboxed)')
    run.add_argument('--full-access', action='store_true', help='Also allow file tools outside the run workspace; requires --allow-shell')
    run.add_argument('--verbose', action='store_true', help='Display full conversations and tool results in terminal')
    run.add_argument('--demo', action='store_true', help='Offline synthetic demonstration; no LLM or API keys required')
    add = commands.add_parser('add-material', help='Import context into an existing stopped run')
    add.add_argument('--run-dir', type=Path, required=True)
    add.add_argument('--proposal', type=Path, action='append', default=[])
    add.add_argument('--material', type=Path, action='append', default=[])
    status = commands.add_parser('status', help='Read saved progress and run summary')
    status.add_argument('run_dir', type=Path)
    doctor = commands.add_parser('doctor', help='Check configuration and local model connectivity')
    doctor.add_argument('--base-url', default=os.getenv('AGEN1_BASE_URL', 'http://127.0.0.1:11434/v1'))
    doctor.add_argument('--model', default=os.getenv('AGEN1_MODEL', 'qwen2.5:14b'))
    return p


async def doctor(args):
    for name in ('SEMANTIC_SCHOLAR_API_KEY', 'BRAVE_SEARCH_API_KEY', 'LOCAL_LLM_API_KEY'):
        print(f'{name}: {"set" if os.getenv(name) else "not set"}')
    headers = {'Authorization': 'Bearer ' + os.environ['LOCAL_LLM_API_KEY']} if os.getenv('LOCAL_LLM_API_KEY') else {}
    data = await HTTP(10).json(args.base_url.rstrip('/') + '/models', headers=headers)
    models = [x['id'] for x in data.get('data', [])]
    print('Available models:', ', '.join(models) or '(none)')
    if args.model not in models:
        raise ValueError(f'Requested model {args.model} is not listed by the server')
    print('Model endpoint is reachable. Research API credentials have not been validated remotely.')


async def run(config):
    journal = Journal(config.run_dir, verbose=config.verbose)
    store = Store(config.run_dir, config.topic, journal)
    await import_materials(store, journal, config.proposal, config.material, timeout=config.request_timeout)
    http = DemoHTTP() if config.demo else HTTP(config.request_timeout)
    llm = LLM(config, journal, http)
    tools = Tools(config, store, journal, http)
    engine = Engine(config, llm, tools, store, journal)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, engine.request_stop)
    try:
        if config.demo:
            journal.event('coordinator', 'demo', 'OFFLINE DEMO: all retrieved evidence is synthetic')
        if config.allow_shell:
            journal.event('coordinator', 'permissions', 'Autonomous shell enabled with current OS-user permissions; this is not a sandbox')
        return await engine.run()
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


@contextmanager
def run_lock(directory):
    import fcntl
    with (directory / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another research process is using this run directory; stop it before importing materials') from None
        yield


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'doctor':
            asyncio.run(doctor(args))
            return 0
        if args.command == 'add-material':
            directory = args.run_dir.expanduser().resolve()
            if not args.proposal and not args.material:
                raise ValueError('Provide at least one --proposal or --material file')
            if not (directory / 'state.json').is_file():
                raise ValueError('Run directory does not exist; use run with --proposal/--material to start it')
            with run_lock(directory):
                topic = json.loads((directory / 'state.json').read_text(encoding='utf-8'))['topic']
                journal = Journal(directory)
                store = Store(directory, topic, journal, new_session=False)
                ids = asyncio.run(import_materials(store, journal, args.proposal, args.material))
            print(f'Imported {len(ids)} new material(s). Resume the run to use them.')
            return 0
        if args.command == 'status':
            for name in ('summary.json', 'status.json', 'plan.json'):
                path = args.run_dir / name
                if path.exists():
                    print(f'{name}:\n{path.read_text()}')
            return 0
        values = vars(args).copy()
        values.pop('command')
        seconds = values.pop('seconds')
        values['hours'] = seconds / 3600 if seconds is not None else (args.hours if args.hours is not None else 1)
        values['run_dir'] = (args.run_dir or Path('runs') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')).resolve()
        existing = values['run_dir'] / 'state.json'
        if not values['topic'] and existing.exists():
            values['topic'] = json.loads(existing.read_text())['topic']
        values['topic'] = values['topic'] or ''
        config = Config(**values)
        config.validate()
        config.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with run_lock(config.run_dir):
            summary = asyncio.run(run(config))
        print(f"Artifacts: {config.run_dir}")
        return 0 if summary['notes'] else 2
    except (ValueError, RuntimeError, OSError, KeyError) as e:
        print(f'Error: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

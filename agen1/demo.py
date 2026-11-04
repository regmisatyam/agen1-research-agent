"""Deterministic offline transport for demonstrations, never a research source."""
import asyncio
import json
import re


class DemoHTTP:
    async def json(self, url, **kwargs):
        await asyncio.sleep(.02)
        if url.endswith('/chat/completions'):
            messages = kwargs['body']['messages']
            system = messages[0]['content']
            if 'You plan literature research' in system:
                count = int(re.search(r'Workers: (\d+)', messages[1]['content']).group(1))
                content = json.dumps({'assignments': [f'DEMO: examine synthetic evidence from perspective {i+1}' for i in range(count)]})
            elif 'Write a rigorous' in system:
                sid = re.search(r'"(S[0-9a-f]{10})"', messages[1]['content']).group(1)
                content = f'# Offline demonstration paper\n\n## Abstract\nThis is synthetic demonstration data, not real research.\n\n## Method\nAn offline transport supplied a fictional source.\n\n## Findings\nThe demo exercised retrieval and note-taking [{sid}].\n\n## Limitations\nNo scientific conclusions may be drawn.\n\n## Conclusion\nThe workflow executed successfully.'
                evidence = json.loads(messages[1]['content'].split('Evidence: ', 1)[1])
                if evidence.get('materials'):
                    content += '\n\n## Supplied context (unverified)\n' + '\n'.join('- ' + m['title'] + ' [' + m['id'] + ']' for m in evidence['materials'])
            else:
                turns = sum(m['role'] == 'assistant' for m in messages)
                context = json.loads(messages[1]['content'].split('Shared evidence:\n', 1)[1])
                supplied = context.get('materials', [])
                if supplied and turns == 0:
                    content = json.dumps({'tool': 'read_material', 'args': {'source_id': supplied[0]['id']}})
                    return {'choices': [{'message': {'content': content}}], 'usage': {'total_tokens': 100}}
                if supplied and turns == 1:
                    task = system.split('Your assignment:')[-1].split('\n')[0]
                    content = json.dumps({'tool': 'note', 'args': {'text': 'Synthetic demo: inspected supplied context (unverified): ' + supplied[0]['title'] + '; ' + task, 'citations': [supplied[0]['id']]}})
                    return {'choices': [{'message': {'content': content}}], 'usage': {'total_tokens': 100}}
                turns -= 2 if supplied else 0
                if turns == 0:
                    content = json.dumps({'tool': 'search_papers', 'args': {'query': 'synthetic demonstration'}})
                elif turns == 1:
                    sid = re.search(r'"(S[0-9a-f]{10})"', messages[-1]['content']).group(1)
                    task = system.split('Your assignment:')[-1].split('\n')[0]
                    content = json.dumps({'tool': 'note', 'args': {'text': 'Synthetic demo finding: ' + task, 'citations': [sid]}})
                else:
                    content = json.dumps({'tool': 'finish', 'args': {'reason': 'Offline demonstration complete'}})
            return {'choices': [{'message': {'content': content}}], 'usage': {'total_tokens': 100}}
        if 'api.semanticscholar.org' in url:
            return {'data': [{'paperId': 'demo-only', 'title': 'Synthetic demonstration source (not a real paper)',
                             'url': 'https://example.invalid/demo', 'abstract': 'Fictional fixture used only to test workflow execution.', 'year': 2026, 'authors': []}]}
        raise RuntimeError('Offline demo blocks all other network destinations')

    async def request(self, url, **kwargs):
        raise RuntimeError('Offline demo does not fetch real web pages or PDFs')

"""One bounded HTTP operation per child process; credentials arrive over stdin."""
import base64
import http.client
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


def read_stream(response, params):
    """Collapse an event stream into one completion, keeping text if the stream breaks."""
    content, reasoning, usage, finish_reason, size = [], [], {}, None, 0
    # Finish before the parent's hard deadline so a long draft is returned, not lost.
    deadline = time.monotonic() + params['timeout'] * .95
    try:
        for raw in response:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            chunk = json.loads(data)
            usage = chunk.get('usage') or usage
            for choice in chunk.get('choices') or []:
                delta = choice.get('delta') or {}
                text, thought = delta.get('content') or '', delta.get('reasoning_content') or delta.get('reasoning') or ''
                content.append(text)
                reasoning.append(thought)
                size += len(text) + len(thought)
                finish_reason = choice.get('finish_reason') or finish_reason
            if size > params['max_bytes']:
                raise OverflowError
            if time.monotonic() > deadline:
                break
    except (http.client.HTTPException, OSError, ValueError):
        if not any(content):
            raise
    # A stream that ended without a finish reason was cut off.
    return {'choices': [{'message': {'role': 'assistant', 'content': ''.join(content), 'reasoning_content': ''.join(reasoning)},
                         'finish_reason': finish_reason or 'interrupted'}], 'usage': usage}


def main():
    params = json.load(sys.stdin)
    url = params['url']
    payload = None if params['body'] is None else json.dumps(params['body']).encode()
    req = urllib.request.Request(url, data=payload, headers={'User-Agent': 'agen1-research/0.1', **params['headers']})
    context = ssl.create_default_context(cafile=os.getenv('SSL_CERT_FILE') or None)
    try:
        with urllib.request.urlopen(req, timeout=params['timeout'], context=context) as response:
            if (params['body'] or {}).get('stream') and 'text/event-stream' in response.headers.get('Content-Type', ''):
                completion = json.dumps(read_stream(response, params)).encode()
                json.dump({'data': base64.b64encode(completion).decode('ascii')}, sys.stdout)
                return
            data = response.read(params['max_bytes'] + 1)
            expected = response.headers.get('Content-Length', '')
            if len(data) > params['max_bytes']:
                result = {'error': 'HTTP response exceeds configured size limit', 'retryable': False}
            # A sized read returns a short body, without raising, if the connection drops.
            elif expected.isdigit() and len(data) < int(expected):
                result = {'error': f'Truncated response from {urlsplit(url).hostname}: {len(data)} of {expected} bytes', 'retryable': True}
            else:
                result = {'data': base64.b64encode(data).decode('ascii')}
    except urllib.error.HTTPError as e:
        try:
            retry_after = min(60, max(0, float(e.headers.get('Retry-After', '0'))))
        except ValueError:
            retry_after = 0
        detail = ''
        # Only the model server receives POSTs, and its error text says why it refused
        # a request. Error bodies from fetched web pages stay out of the prompts.
        if params['body'] is not None and 400 <= e.code < 500:
            try:
                detail = ': ' + e.read(2000).decode('utf-8', 'replace')
            except (http.client.HTTPException, OSError):
                pass
        result = {'error': f'HTTP {e.code} from {urlsplit(url).hostname}{detail}', 'retryable': e.code == 429 or e.code >= 500, 'retry_after': retry_after}
    except OverflowError:
        result = {'error': 'HTTP response exceeds configured size limit', 'retryable': False}
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError, ValueError) as e:
        result = {'error': f'Connection failed to {urlsplit(url).hostname}: {type(e).__name__}', 'retryable': True}
    json.dump(result, sys.stdout)


if __name__ == '__main__':
    main()

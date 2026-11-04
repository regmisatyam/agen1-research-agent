"""One bounded HTTP operation per child process; credentials arrive over stdin."""
import base64
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit


def main():
    params = json.load(sys.stdin)
    url = params['url']
    payload = None if params['body'] is None else json.dumps(params['body']).encode()
    req = urllib.request.Request(url, data=payload, headers={'User-Agent': 'agen1-research/0.1', **params['headers']})
    context = ssl.create_default_context(cafile=os.getenv('SSL_CERT_FILE') or None)
    try:
        with urllib.request.urlopen(req, timeout=params['timeout'], context=context) as response:
            data = response.read(params['max_bytes'] + 1)
            if len(data) > params['max_bytes']:
                result = {'error': 'HTTP response exceeds configured size limit', 'retryable': False}
            else:
                result = {'data': base64.b64encode(data).decode('ascii')}
    except urllib.error.HTTPError as e:
        try:
            retry_after = min(60, max(0, float(e.headers.get('Retry-After', '0'))))
        except ValueError:
            retry_after = 0
        result = {'error': f'HTTP {e.code} from {urlsplit(url).hostname}', 'retryable': e.code == 429 or e.code >= 500, 'retry_after': retry_after}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        result = {'error': f'Connection failed to {urlsplit(url).hostname}: {type(e).__name__}', 'retryable': True}
    json.dump(result, sys.stdout)


if __name__ == '__main__':
    main()

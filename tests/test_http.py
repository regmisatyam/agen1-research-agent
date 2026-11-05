import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
import unittest
from unittest.mock import patch

from agen1.network import HTTP, HTTPError


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reject(self):
        self.send_response(400)
        self.end_headers()
        self.wfile.write(b'maximum context length is 5000 tokens')

    def do_GET(self):
        if self.path == '/slow':
            time.sleep(1)
        if self.path == '/reject':
            return self.reject()
        if self.path == '/rate':
            self.send_response(429)
            self.send_header('Retry-After', '2')
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        try:
            self.wfile.write(json.dumps({'data': [{'id': 'local-test-model'}]}).encode())
        except BrokenPipeError:
            pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/reject':
            return self.reject()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({'echo': payload}).encode())


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        # Only loopback is bypassed for the local test fixture; Internet proxy stays intact.
        self.patch = patch.dict(os.environ, {'NO_PROXY': '127.0.0.1', 'no_proxy': '127.0.0.1'})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    async def test_real_http_get_and_post(self):
        client = HTTP(3)
        data = await client.json(self.url + '/models')
        self.assertEqual(data['data'][0]['id'], 'local-test-model')
        result = await client.json(self.url, headers={'Content-Type': 'application/json'}, body={'messages': ['hello']})
        self.assertEqual(result['echo']['messages'], ['hello'])

    async def test_rate_limit_metadata(self):
        with self.assertRaises(HTTPError) as ctx:
            await HTTP(3).json(self.url + '/rate')
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(ctx.exception.retry_after, 2)

    async def test_only_the_model_servers_error_text_is_kept(self):
        with self.assertRaises(HTTPError) as posted:
            await HTTP(3).json(self.url + '/reject', headers={'Content-Type': 'application/json'}, body={'messages': []})
        self.assertIn('HTTP 400', str(posted.exception))
        self.assertIn('maximum context length is 5000', str(posted.exception))
        # Error pages from fetched sites are untrusted and must not reach a prompt.
        with self.assertRaises(HTTPError) as fetched:
            await HTTP(3).request(self.url + '/reject')
        self.assertIn('HTTP 400', str(fetched.exception))
        self.assertNotIn('maximum context length', str(fetched.exception))

    async def test_hard_timeout_and_size_limit(self):
        start = time.monotonic()
        with self.assertRaises(HTTPError):
            await HTTP(.2).json(self.url + '/slow')
        self.assertLess(time.monotonic() - start, .8)
        with self.assertRaises(HTTPError):
            await HTTP(3).request(self.url, max_bytes=2)

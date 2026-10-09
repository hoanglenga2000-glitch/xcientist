"""Private stdin/stdout HTTP worker. It never writes request/response files."""
import json
import importlib.util
from pathlib import Path
import sys
import urllib.error
import urllib.request

MAX_BYTES = 16 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError('model_redirect_rejected')


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('request_size_exceeded')
        request = json.loads(raw)
        if request.get('stream') is True:
            spec = importlib.util.spec_from_file_location('chat_stream_transport', Path(__file__).with_name('chat_stream_transport.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            def emit_text(text):
                print(json.dumps({'kind': 'text', 'text': text}, ensure_ascii=True), flush=True)
            body = module.post_chat_stream(request['url'], request['headers'], request['payload'], request['timeout'], emit_text)
            print(json.dumps({'kind': 'response', 'body': body}, ensure_ascii=True), flush=True)
            return
        headers = {'User-Agent': 'research-os-evolution/1.0', **request['headers']}
        req = urllib.request.Request(request['url'], data=json.dumps(request['payload']).encode('utf-8'), headers=headers, method='POST')
        opener = urllib.request.build_opener(NoRedirect())
        with opener.open(req, timeout=request['timeout']) as response:
            body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ValueError('response_size_exceeded')
        result = {'kind': 'response', 'body': json.loads(body.decode('utf-8'))}
    except urllib.error.HTTPError as error:
        result = {'kind': 'http_error', 'status': int(error.code), 'retry_after': error.headers.get('Retry-After')}
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        result = {'kind': 'network_error'}
    except Exception as error:
        result = {'kind': 'protocol_error'}
        if str(error) == 'stream_model_identity_invalid':
            result['code'] = 'model_identity_unconfirmed'
    sys.stdout.write(json.dumps(result, ensure_ascii=True) + '\n')


if __name__ == '__main__':
    main()

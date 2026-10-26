"""One public HTTPS request, DNS-pinned, no proxy/redirect/retry, bounded worker."""
from __future__ import annotations

import http.client
import ipaddress
import json
import re
import socket
import ssl
import sys
from urllib.parse import urlsplit


def safe_url(value):
    if not isinstance(value, str) or len(value) > 500 or re.search(r'[\s\\\x00-\x1f]', value):
        raise ValueError('model_address_invalid')
    try:
        url = urlsplit(value)
        host = url.hostname or ''
        if (url.scheme != 'https' or url.port not in (None, 443) or url.username or url.password
                or url.query or url.fragment or not re.fullmatch(r'[a-zA-Z0-9.-]+', host)
                or '.' not in host or host.endswith('.') or '..' in host
                or any(host.endswith('.' + ending) for ending in ('local', 'localhost', 'internal', 'test', 'invalid'))):
            raise ValueError()
        # IP literals and exotic numeric forms are not supported as user endpoints.
        if not re.fullmatch(r'[a-zA-Z]{2,63}', host.rsplit('.', 1)[-1]):
            raise ValueError()
        if any(not re.fullmatch(r'[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?', label) for label in host.split('.')):
            raise ValueError()
    except ValueError:
        raise ValueError('model_address_invalid') from None
    return url


def public_addresses(host):
    addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    if not addresses or len(addresses) > 64:
        raise ValueError('model_address_resolution_failed')
    for family, kind, proto, canon, address in addresses:
        ip = ipaddress.ip_address(address[0])
        if (not ip.is_global or ip.is_multicast or ip.is_reserved
                or (ip.version == 6 and (ip not in ipaddress.ip_network('2000::/3') or ip.sixtofour or ip.teredo))):
            raise ValueError('model_address_not_public')
    return addresses


RESPONSE_BYTE_CAP = 1024 * 1024


class SseAssembler:
    """Join one SSE event without retaining more than the personal response cap."""

    def __init__(self, limit=RESPONSE_BYTE_CAP):
        self.limit, self.total, self.pending = limit, 0, []

    def feed(self, raw):
        if not isinstance(raw, (bytes, bytearray)):
            raise ValueError('model_response_invalid')
        self.total += len(raw)
        if len(raw) > self.limit or self.total > self.limit:
            raise ValueError('model_response_too_large')
        try:
            line = raw.decode('utf-8').rstrip('\r\n')
        except UnicodeDecodeError:
            raise ValueError('model_response_invalid') from None
        if line == '':
            if not self.pending:
                return None
            data = '\n'.join(self.pending)
            self.pending = []
            return data
        if line.startswith('data:'):
            self.pending.append(line[5:].lstrip())
            return None
        if line.startswith(('event:', 'id:', 'retry:', ':')):
            return None
        raise ValueError('model_response_invalid')

    def finish(self):
        if not self.pending:
            return None
        data = '\n'.join(self.pending)
        self.pending = []
        return data


def _exchange(request):
    """Pinned public POST. Caller reads the body and closes both objects."""
    url = safe_url(request['url'])
    addresses = public_addresses(url.hostname)
    family, kind, proto, canon, address = addresses[0]
    context = ssl.create_default_context()
    connection = http.client.HTTPSConnection(url.hostname, timeout=15, context=context)
    # Connect directly to the validated IP; TLS still verifies the original hostname.
    sock = socket.socket(family, kind, proto)
    sock.settimeout(15)
    try:
        sock.connect(address)
        connection.sock = context.wrap_socket(sock, server_hostname=url.hostname)
        payload = json.dumps(request['payload']).encode()
        if len(payload) > 2 * 1024 * 1024:
            raise ValueError('model_request_too_large')
        connection.request('POST', url.path or '/', body=payload, headers=request['headers'])
        response = connection.getresponse()
        if 300 <= response.status < 400:
            raise ValueError('model_redirect_rejected')
        if not 200 <= response.status < 300:
            raise ValueError('model_http_' + str(response.status))
        return connection, sock, response
    except Exception:
        connection.close()
        sock.close()
        raise


def request(request):
    connection, sock, response = _exchange(request)
    try:
        body = response.read(RESPONSE_BYTE_CAP + 1)
        if len(body) > RESPONSE_BYTE_CAP:
            raise ValueError('model_response_too_large')
        return json.loads(body)
    finally:
        connection.close()
        sock.close()


def stream_request(request):
    """Print one JSON object per SSE data event. The parent reconstructs the turn."""
    connection, sock, response = _exchange(request)
    try:
        assembler = SseAssembler()
        while True:
            raw = response.readline(RESPONSE_BYTE_CAP + 1)
            if not raw:
                break
            data = assembler.feed(raw)
            if data is not None:
                print(json.dumps({'data': data}, ensure_ascii=False), flush=True)
        tail = assembler.finish()
        if tail is not None:
            print(json.dumps({'data': tail}, ensure_ascii=False), flush=True)
    finally:
        connection.close()
        sock.close()


def public_error(error):
    code = str(error)
    return {'error': code if re.fullmatch(r'model_[a-z0-9_]{1,80}', code) else 'model_network_failed'}


if __name__ == '__main__':
    try:
        raw = sys.stdin.buffer.read(3 * 1024 * 1024 + 1)
        if len(raw) > 3 * 1024 * 1024:
            raise ValueError('model_request_too_large')
        incoming = json.loads(raw)
        if incoming.get('stream') is True:
            stream_request(incoming)
        else:
            print(json.dumps({'body': request(incoming)}), flush=True)
    except Exception as error:
        print(json.dumps(public_error(error)), flush=True)

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


def request(request):
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
        body = response.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise ValueError('model_response_too_large')
        return json.loads(body)
    finally:
        connection.close()
        sock.close()


if __name__ == '__main__':
    try:
        raw = sys.stdin.buffer.read(3 * 1024 * 1024 + 1)
        if len(raw) > 3 * 1024 * 1024:
            raise ValueError('model_request_too_large')
        result = {'body': request(json.loads(raw))}
    except Exception as error:
        code = str(error)
        result = {'error': code if re.fullmatch(r'model_[a-z0-9_]{1,80}', code) else 'model_network_failed'}
    print(json.dumps(result))

"""External transport safety; resolver/socket doubles are not real service validation."""
import socket
import pytest
from evomind_runtime.personal_model_http import public_addresses, safe_url
from evomind_runtime.personal_model_client import post_public_json
from evomind_runtime.model_profile_secrets import crypt


@pytest.mark.parametrize('address', ['127.0.0.1', '10.20.30.40', '169.254.169.254', '100.64.0.1', '::1', 'fc00::1', '64:ff9b::a00:1', '2002:0a00:0001::'])
def test_any_nonpublic_dns_answer_blocks_entire_destination(monkeypatch, address):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.215.14', 443)),
        (socket.AF_INET6 if ':' in address else socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 443)),
    ])
    with pytest.raises(ValueError, match='model_address_not_public'):
        public_addresses('api.example.com')


def test_isolated_process_cannot_send_credentials_to_an_external_endpoint(monkeypatch):
    monkeypatch.setenv('EVOMIND_TEST_FIXTURE_ROOT', 'isolated-test')
    with pytest.raises(ValueError, match='model_isolated_network_disabled'):
        post_public_json('https://api.example.com', {'Authorization': 'Bearer fixture'}, {})


def test_dpapi_ciphertext_is_bound_to_owner_and_revision():
    plaintext = b'public-fixture-credential'
    cipher = crypt(plaintext, 'tenant/alice/model/version-one')
    assert plaintext not in cipher
    assert crypt(cipher, 'tenant/alice/model/version-one', decrypt=True) == plaintext
    with pytest.raises(ValueError, match='credential_protection_failed'):
        crypt(cipher, 'tenant/bob/model/version-one', decrypt=True)
    with pytest.raises(ValueError, match='credential_protection_failed'):
        crypt(cipher, 'tenant/alice/model/version-two', decrypt=True)

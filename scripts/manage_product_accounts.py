"""Explicit, offline account administration. Never called by release transactions.

Passwords are read without echo (or --password-stdin for isolated fixtures).
Run as the intended service identity; existing directory ACLs are only checked.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
from pathlib import Path
import re
import secrets
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from evomind_runtime.model_profile_secrets import private_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['create', 'disable', 'enable', 'reset-password'])
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--username', required=True)
    parser.add_argument('--role', choices=['user', 'admin'], default='user')
    parser.add_argument('--password-stdin', action='store_true')
    parser.add_argument('--confirm-account-change', action='store_true', required=True)
    args = parser.parse_args()
    if not args.database.is_absolute() or not re.fullmatch(r'[A-Za-z0-9._-]{1,64}', args.username):
        raise ValueError('invalid_account_target')
    private_directory(args.database.parent)
    if args.database.exists() and getattr(args.database.lstat(), 'st_file_attributes', 0) & 1024:
        raise ValueError('account_database_alias_rejected')
    encoded = None
    if args.action in {'create', 'reset-password'}:
        password = sys.stdin.readline().rstrip('\r\n') if args.password_stdin else getpass.getpass('New password: ')
        if not 12 <= len(password) <= 256:
            raise ValueError('password_requires_12_to_256_characters')
        salt = secrets.token_bytes(16)
        digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32).hex()
        encoded = f'scrypt$16384$8$1${salt.hex()}${digest}'
        del password
    if not args.database.exists() and args.action != 'create':
        raise ValueError('account_database_missing')
    with sqlite3.connect(args.database) as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('''CREATE TABLE IF NOT EXISTS accounts (username TEXT PRIMARY KEY COLLATE NOCASE,
            tenant_id TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','admin')),
            enabled INTEGER NOT NULL CHECK(enabled IN (0,1)), session_version INTEGER NOT NULL)''')
        if args.action == 'create':
            tenant = 'tenant_' + hashlib.sha256(('evomind.tenant.v1:' + args.username.lower()).encode()).hexdigest()[:24]
            db.execute('INSERT INTO accounts VALUES (?,?,?,?,1,1)', (args.username, tenant, encoded, args.role))
        else:
            row = db.execute('SELECT username FROM accounts WHERE username=? COLLATE NOCASE', (args.username,)).fetchone()
            if not row:
                raise ValueError('account_not_found')
            if args.action == 'reset-password':
                db.execute('UPDATE accounts SET password_hash=?, session_version=session_version+1 WHERE username=?', (encoded, row[0]))
            else:
                db.execute('UPDATE accounts SET enabled=?, session_version=session_version+1 WHERE username=?', (int(args.action == 'enable'), row[0]))
    print(json.dumps({'ok': True, 'action': args.action, 'username': args.username}))


if __name__ == '__main__':
    main()

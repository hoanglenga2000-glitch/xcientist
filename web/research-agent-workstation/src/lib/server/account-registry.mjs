import { isAbsolute } from 'node:path';
import { lstatSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';

// Provisioned offline in a restricted directory. Web requests never create,
// migrate or fall back from this database to the legacy shared login.
export function accountRegistryEnabled() {
  return Boolean(process.env.WORKSTATION_ACCOUNTS_DB);
}

export function findAccount(username) {
  const path = process.env.WORKSTATION_ACCOUNTS_DB;
  if (!path) return null;
  if (!isAbsolute(path) || !lstatSync(path).isFile() || lstatSync(path).isSymbolicLink()) {
    throw new Error('account_registry_unavailable');
  }
  const db = new DatabaseSync(path, { readOnly: true, allowExtension: false });
  try {
    const row = db.prepare('SELECT username, tenant_id, password_hash, role, enabled, session_version FROM accounts WHERE username = ? COLLATE NOCASE').get(username);
    if (!row) return null;
    if (!/^[A-Za-z0-9._-]{1,64}$/.test(row.username) || !/^tenant_[a-f0-9]{24}$/.test(row.tenant_id)
      || !['user', 'admin'].includes(row.role) || ![0, 1].includes(row.enabled)
      || !Number.isSafeInteger(row.session_version) || row.session_version < 1
      || !/^scrypt\$16384\$8\$1\$[a-f0-9]{32}\$[a-f0-9]{64}$/.test(row.password_hash)) {
      throw new Error('account_registry_invalid');
    }
    return row;
  } finally { db.close(); }
}

import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { preferenceKey, validatePreferences, mergePreferences } from './user-preferences.ts';
const alice = {username:'alice',tenantId:`tenant_${'a'.repeat(24)}`,authentication:'public_password'};
test('preference keys isolate same-tenant users and distinct tenants', () => {
  const keys = [alice, {...alice,username:'bob'}, {...alice,tenantId:`tenant_${'b'.repeat(24)}`}].map(preferenceKey);
  assert.equal(new Set(keys).size,3);
  assert.equal(preferenceKey(alice), keys[0]);
  assert.throws(()=>preferenceKey(null));
  assert.throws(()=>preferenceKey({...alice,tenantId:null}));
});
test('preference allowlist rejects arbitrary configuration, secrets and prototype keys', () => {
  for (const value of [{},{general:{root:'C:/'}},{credentials:{api_key:'test'}},{general:{theme:'other'}},[],JSON.parse('{"__proto__":{"theme":"dark"}}'),{language:{ui_language:[]}}]) {
    assert.throws(()=>validatePreferences(value));
  }
  assert.deepEqual(validatePreferences({general:{theme:'light'}}),{general:{theme:'light'}});
});
test('partial updates preserve unrelated preferences without leaking global values', () => {
  const old=mergePreferences({general:{theme:'light'},language:{ui_language:'en-US'}});
  const next=mergePreferences(old,{language:{report_language:'en-US'}});
  assert.equal(next.general.theme,'light'); assert.equal(next.language.ui_language,'en-US');
  assert.equal(next.language.report_language,'en-US');
  assert.equal(JSON.stringify(mergePreferences({credentials:{token:'do-not-return'}})).includes('do-not-return'),false);
});
test('settings endpoint uses authenticated scoped key, atomic update and no global enumeration', async () => {
  const route=await readFile('src/app/api/settings/route.ts','utf8');
  assert.match(route,/sessionPrincipal/); assert.match(route,/preferenceKey/); assert.match(route,/\$transaction/);
  assert.doesNotMatch(route,/findMany|logAction|ensureWorkstationSeeded/);
});

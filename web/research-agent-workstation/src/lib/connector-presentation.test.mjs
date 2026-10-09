import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { connectorStatus, connectorLabel, preferredModelConnector, hpcConnectionVerified, connectorHealthReady } from './connector-presentation.ts';
test('missing connector evidence is unknown, not absent credentials', () => {
  assert.equal(connectorLabel('zh-CN', connectorStatus(null, 'gpu')), '未知');
  assert.equal(connectorStatus({connector_status:{gpu:{configured:'false'}}}, 'gpu').configured, null);
});
test('HPC connection requires fresh complete container and proxy identity evidence', () => {
  const entry = {configured:true,current_gate_ready:true,job_container_verified:true,designated_proxy_path_verified:true,identity_samples_passed:5,live_probe_fresh:true,proof_expires_at:'2026-09-09T00:00:00Z'};
  assert.equal(hpcConnectionVerified(entry,Date.parse('2026-09-08T00:00:00Z')),true);
  for (const key of Object.keys(entry)) { const missing={...entry}; delete missing[key]; assert.equal(hpcConnectionVerified(missing,Date.parse('2026-09-08T00:00:00Z')),false); }
  assert.equal(hpcConnectionVerified(entry,Date.parse('2026-09-10T00:00:00Z')),false);
  assert.equal(hpcConnectionVerified({configured:true,state:'DEGRADED'}),false);
});
test('negative and stale states cannot match a positive substring', () => {
  for (const state of ['unverified', 'not_ready', 'failed', 'expired', 'blocked_ready']) {
    assert.notEqual(connectorStatus({connector_status:{gpu:{configured:true, state}}}, 'gpu').tone, 'verified');
  }
});

test('GPU sidebar and overview never call an expired identity proof ready', () => {
  const view = connectorStatus({connector_status:{gpu:{configured:true,state:'ready',current_gate_ready:false,live_probe_fresh:false,proof_expires_at:'2000-01-01T00:00:00Z'}}}, 'gpu');
  assert.equal(view.tone, 'stale');
  assert.equal(connectorLabel('zh-CN', view), '待重新核验');
});
test('model presentation follows real configured model and exact ready evidence', () => {
  const summary = {connector_status:{openai:{configured:true,state:'ready',model:'gpt-5.5'},deepseek:{configured:false,model:'deepseek'}}};
  assert.deepEqual(preferredModelConnector(summary), {key:'openai',model:'gpt-5.5'});
  assert.equal(connectorLabel('zh-CN',connectorStatus(summary,'openai')), '就绪');
});

test('overall connector readiness requires current GPU, Kaggle, and model evidence', () => {
  const summary = {connector_status:{gpu:{configured:true,state:'ready',current_gate_ready:true,job_container_verified:true,designated_proxy_path_verified:true,identity_samples_passed:5,live_probe_fresh:true,proof_expires_at:'2099-01-01T00:00:00Z'},kaggle:{configured:true,state:'ready'},llm:{configured:true,state:'ready',model:'deepseek-flash'}}};
  assert.equal(connectorHealthReady(summary), true);
  for (const key of ['gpu','kaggle','llm']) {
    const missing = structuredClone(summary); delete missing.connector_status[key];
    assert.equal(connectorHealthReady(missing), false);
    for (const state of ['failed','expired','not_ready','unverified']) {
      const stale = structuredClone(summary); stale.connector_status[key].state = state;
      assert.equal(connectorHealthReady(stale), false);
    }
  }
});

test('overview and settings consume the same canonical connector projection as sidebar', async () => {
  for (const name of ['OverviewScreen.tsx','SettingsScreen.tsx']) {
    const source = await readFile(new URL('../components/workstation/screens/' + name, import.meta.url), 'utf8');
    assert.match(source, /@\/lib\/connector-presentation/);
    assert.match(source, /preferredModelConnector\(summary\)/);
    assert.doesNotMatch(source, /key: "(?:local_hpc|gpu_hpc|kaggle_api|deepseek_api)"/);
    assert.doesNotMatch(source, /recordSettingsAction\("test_all_connectors"\)/);
  }
  const overview = await readFile(new URL('../components/workstation/screens/OverviewScreen.tsx', import.meta.url), 'utf8');
  assert.match(overview, /connectorHealthReady\(summary\)/);
  assert.doesNotMatch(overview, /"System Health"/);
});

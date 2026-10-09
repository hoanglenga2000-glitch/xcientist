import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const preview = path.dirname(fileURLToPath(import.meta.url));
const web = path.resolve(preview, '../..');
const repo = path.resolve(web, '../..');
const output = path.join(repo, 'artifacts/advanced-tools-usability-20261008');
const baseline = 'D:/EV12/system-repair-20261008-gates';
await mkdir(output, { recursive: true });
const hash = async filename => createHash('sha256').update(await readFile(filename)).digest('hex');
const receiptHash = await hash(path.join(baseline, 'source-receipt.json'));
if (receiptHash !== '4738be6fffac739a6d3f3435b933242865d529cd3caba1134c9070e7bd924d1e') throw Error('Published source receipt changed; re-audit before proceeding.');
const compared = [
  'src/app/page.tsx', 'src/app/layout.tsx', 'src/components/workstation/navigation.ts',
  'src/components/workstation/Sidebar.tsx', 'src/components/workstation/AppShell.tsx',
  ...['OverviewScreen','TasksScreen','ExperimentsScreen','LiteratureScreen','CodeAgentScreen','GpuHpcScreen','WorkflowScreen','RuntimeScreen','EvidenceLedgerScreen','GatesScreen','SettingsScreen','ProjectsScreen','AssistantScreen'].map(name => `src/components/workstation/screens/${name}.tsx`),
  'src/components/workstation/AiControlConsole.tsx', 'src/components/workstation/EvolutionConsole.tsx',
  'src/app/api/tasks/route.ts', 'src/app/api/settings/route.ts', 'src/lib/server/scoped-runtime-fetch.ts', 'src/lib/server/user-preferences.ts', 'package.json', 'package-lock.json',
];
const comparison = [];
for (const name of compared) {
  try {
    const local = await hash(path.join(web, name));
    const published = await hash(path.join(baseline, 'web', name));
    comparison.push({ path: name, published_sha256: published, local_sha256: local, local_differs: local !== published, included_in_preview_patch: false });
  } catch (error) {
    const localHash = await hash(path.join(web, name)).catch(() => null);
    comparison.push({ path: name, published_sha256: null, local_sha256: localHash, local_differs: !!localHash, checked: !!localHash, reason: localHash ? 'not_in_published_snapshot' : error.code, included_in_preview_patch: false });
  }
}
const candidate = [];
async function collect(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (['.next','node_modules','next-env.d.ts'].includes(entry.name) || entry.name.endsWith('.tsbuildinfo')) continue;
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) await collect(filename);
    else candidate.push({ path: path.relative(repo, filename).replaceAll('\\','/'), sha256: await hash(filename), baseline: 'new_preview_file' });
  }
}
await collect(preview);
candidate.push({ path: 'scripts/start_advanced_tools_preview.ps1', sha256: await hash(path.join(repo,'scripts/start_advanced_tools_preview.ps1')), baseline: 'new_preview_file' });
await writeFile(path.join(output, 'source-scope.json'), JSON.stringify({ scope: 'local_preview_only_not_release_candidate', source_receipt_sha256: receiptHash, published_build: 'overlay-invitation-beta-5141838f452e', comparison, candidate }, null, 2));
const clean = Object.fromEntries(Object.entries(process.env).filter(([key]) => !/(TOKEN|SECRET|PASSWORD|API_KEY|DATABASE_URL|WORKSTATION|EVOMIND|ANTHROPIC|OPENAI|DEEPSEEK|HPC|KAGGLE|CLAUDE)/i.test(key)));
const commands = [
  ['preview-tests', ['--test', 'preview/advanced-tools/tests/api.test.mjs', 'preview/advanced-tools/tests/ui.test.mjs']],
  ['preview-typecheck', ['node_modules/typescript/bin/tsc', '--project', 'preview/advanced-tools/tsconfig.json', '--noEmit', '--incremental', 'false']],
  ['workstation-typecheck', ['node_modules/typescript/bin/tsc', '--noEmit', '--incremental', 'false']],
  ['existing-status-regression', ['--test', 'src/components/workstation/status-truthfulness.test.mjs']],
];
const checks = [];
for (const [name, args] of commands) {
  const result = spawnSync(process.execPath, args, { cwd: web, env: clean, encoding: 'utf8', timeout: 120000, windowsHide: true });
  await writeFile(path.join(output, `${name}.log`), `${result.stdout ?? ''}${result.stderr ?? ''}`);
  checks.push({ name, exit_code: result.status, error: result.error?.message ?? null });
}
await writeFile(path.join(output, 'checks.json'), JSON.stringify({ boundary: 'isolated_sample_UI_API_artifact_only', checks }, null, 2));
console.log(JSON.stringify({ checks, compared: comparison.length, different: comparison.filter(row => row.local_differs).length, missing: comparison.filter(row => row.checked === false), candidate_files: candidate.length, evidence: output }, null, 2));
if (checks.some(check => check.exit_code !== 0)) process.exitCode = 1;

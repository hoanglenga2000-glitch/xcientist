// Acceptance of the exact deployed compiled route; no cookies or secrets.
const fs = require('node:fs');
const path = require('node:path');
const root = 'C:/ProgramData/EvoMind';
const anchorPath = path.join(root, 'state/service-clock-anchor.json');
const anchor = JSON.parse(fs.readFileSync(anchorPath, 'utf8').replace(/^\uFEFF/, ''));
const build = process.argv[2];
if (!/^overlay-invitation-beta-[a-f0-9]{12}$/.test(build)) throw Error('INVALID_BUILD');
Object.assign(process.env, { NODE_ENV: 'production', EVOMIND_SERVICE_CLOCK_ANCHOR_PATH: anchorPath, EVOMIND_SERVICE_INSTANCE_ID: anchor.service_instance_id, EVOMIND_BOOT_ID: anchor.boot_id, EVOMIND_CLOCK_QPC_TIMESTAMP: String(anchor.qpc_timestamp), EVOMIND_CLOCK_QPC_FREQUENCY: String(anchor.qpc_frequency), EVOMIND_CLOCK_EPOCH_MS: String(anchor.windows_utc_epoch_ms) });
const route = require(path.join(root, 'web-overlays', build, '.next/server/app/api/hpc/byoa/enrollment-envelope/route.js'));
(async () => {
  const response = await route.routeModule.userland.GET();
  const result = await response.json();
  if (response.status !== 200 || !result.ok || result.time_source !== 'managed_anchor' || !anchor.renewal) throw Error('COMPILED_CLOCK_CONSUMER_REJECTED');
  console.log(JSON.stringify({ status: 'passed', validation_surface: 'deployed_compiled_route_in_separate_process', build_id: build, renewal_present: true, time_source: result.time_source, utc_epoch_ms: result.utc_epoch_ms, utc_drift_ms: Math.abs(result.utc_epoch_ms - Date.now()) }));
})().catch(error => { console.error(error.message); process.exitCode = 1; });

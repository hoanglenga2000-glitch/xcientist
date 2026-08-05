import { readFileSync } from "node:fs";
import path from "node:path";
import {
  doctor,
  installRelease,
  layout,
  lifecycle,
  openWorkstation,
  recoverReleaseState,
  rollback,
  uninstall,
  verifyReleaseEnvelope,
} from "./core.mjs";

const HELP = `EvoMind Windows CLI

Usage:
  evomind install [--manifest <URL|PATH>] [--version <SEMVER>] [--json]
  evomind start|stop|status|open [--json]
  evomind doctor [--json]
  evomind upgrade [--manifest <URL|PATH>] [--version <SEMVER>] [--json]
  evomind rollback [--no-restart] [--json]
  evomind uninstall [--purge-data] [--json]
`;

function cliLayout() {
  const capabilityPath = process.env.EVOMIND_QA_LAYOUT_CAPABILITY;
  const nonce = process.env.EVOMIND_QA_LAYOUT_NONCE;
  if (!capabilityPath && !nonce) return layout();
  if (!capabilityPath || !nonce || !/^[A-Za-z0-9_-]{32,128}$/.test(nonce)) {
    throw new Error("QA layout capability is incomplete");
  }
  const absolute = path.resolve(capabilityPath);
  let capability;
  try { capability = JSON.parse(readFileSync(absolute, "utf8")); }
  catch { throw new Error("QA layout capability could not be read"); }
  const capabilityRoot = path.dirname(absolute);
  const expectedLocal = path.join(capabilityRoot, "local");
  const expectedRoaming = path.join(capabilityRoot, "roaming");
  if (
    capability?.schema !== "evomind.qa_layout_capability.v1"
    || capability.nonce !== nonce
    || !path.isAbsolute(String(capability.local_base || ""))
    || !path.isAbsolute(String(capability.roaming_base || ""))
    || !path.basename(capabilityRoot).startsWith("evomind-release-smoke-")
    || path.resolve(capability.local_base) !== expectedLocal
    || path.resolve(capability.roaming_base) !== expectedRoaming
  ) throw new Error("QA layout capability is invalid");
  return layout({
    EVOMIND_LOCALAPPDATA: path.resolve(capability.local_base),
    EVOMIND_APPDATA: path.resolve(capability.roaming_base),
  });
}

function parse(argv) {
  const command = argv[0] || "help";
  const options = {};
  for (let index = 1; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--json") options.json = true;
    else if (value === "--purge-data") options.purgeData = true;
    else if (value === "--no-restart") options.restart = false;
    else if (["--manifest", "--version", "--bundle"].includes(value)) {
      const next = argv[index + 1];
      if (!next || next.startsWith("--")) throw new Error(`${value} requires a value`);
      options[value.slice(2)] = next;
      index += 1;
    } else throw new Error(`unknown option: ${value}`);
  }
  return { command, options };
}

function output(result, json) {
  if (json || typeof result !== "string") process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  else process.stdout.write(`${result}\n`);
}

export async function main(argv) {
  const { command, options } = parse(argv);
  if (["help", "--help", "-h"].includes(command)) { process.stdout.write(HELP); return 0; }
  if (command === "version" || command === "--version") { process.stdout.write("0.3.0\n"); return 0; }
  let result;
  if (command === "verify-release") {
    result = await verifyReleaseEnvelope({ manifestSource: options.manifest, archivePath: options.bundle });
    output(result, options.json);
    return result?.ok === true ? 0 : 1;
  }
  const paths = cliLayout();
  if (command === "install" || command === "upgrade") result = await installRelease({ manifestSource: options.manifest, version: options.version, paths });
  else if (["start", "stop", "status"].includes(command)) result = await lifecycle(command, { paths });
  else if (command === "open") result = await openWorkstation(paths);
  else if (command === "doctor") result = await doctor(paths);
  else if (command === "rollback") result = await rollback({ paths, restart: options.restart !== false });
  else if (command === "uninstall") result = await uninstall({ paths, purgeData: options.purgeData === true });
  else if (command === "recover") result = await recoverReleaseState(paths);
  else throw new Error(`unknown command: ${command}`);
  output(result, options.json);
  return result?.ok === false ? 1 : 0;
}

import { execFile } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import { prisma } from "@/lib/db";
import {
  requireNetworkHost,
  requireOptionalProxyUsername,
  requireShellSafePath,
  requireSshUsername,
  requireTcpPort
} from "@/lib/security/network-boundary";
import { logAction } from "@/lib/server/actions";
import { gpuSshConfig, gpuSshStatus, hasGpuSshConfig } from "@/lib/server/capabilities";
import { normalizeTaskId, stamp, workspaceRoot, writeJsonArtifact, writeTextArtifact } from "@/lib/server/paths";
import { evaluateStrategyExecutionGate } from "@/lib/server/strategy-registry";
import { artifactDescriptor, writeArtifactManifestArtifact } from "@/lib/server/workstation-run-contract";

const execFileAsync = promisify(execFile);

export type GpuGatewayResult = {
  ok: true;
  configured: boolean;
  status: "not_configured" | "passed" | "failed" | "submitted" | "cancel_requested" | "rejected" | "blocked_dependency" | "blocked_resource_gateway";
  provider: "ssh_gateway";
  missing_env?: string[];
  host?: string;
  username?: string;
  remote_workspace?: string;
  artifact_path?: string;
  stdout_artifact?: string;
  stderr_artifact?: string;
  job_manifest_path?: string;
  artifact_manifest_path?: string;
  pulled_artifacts?: string[];
  metrics_artifact?: string;
  submission_artifact?: string;
  probabilities_artifact?: string;
  report_artifact?: string;
  allowed_templates?: string[];
  required_fields?: string[];
  stdout?: string;
  stderr?: string;
  error?: string;
};

function shellSingleQuote(value: string) {
  return `'${value.replace(/'/g, "'\\''")}'`;
}

function s6e6BoostingPythonPath() {
  const root = gpuSshConfig().remoteWorkspace.replace(/\/+$/, "");
  return `${root}/pyenvs/s6e6_boosting/bin/python`;
}

async function updateGpuJobManifestStatus(
  manifestPath: string,
  status: "prepared" | "running" | "submitted" | "failed",
  extra: Record<string, unknown> = {}
) {
  const absolutePath = path.join(workspaceRoot, manifestPath);
  const existingText = await fs.promises.readFile(absolutePath, "utf-8").catch(() => "");
  let existing: Record<string, unknown> = {};
  try {
    existing = existingText ? JSON.parse(existingText) as Record<string, unknown> : {};
  } catch {
    existing = {};
  }
  await writeJsonArtifact(manifestPath, {
    ...existing,
    ...extra,
    status,
    updated_at: new Date().toISOString()
  });
}

const allowedTemplates = {
  connection_smoke: gpuTorchSmokeCommand("connection_smoke"),
  house_prices_baseline: gpuTorchSmokeCommand("house_prices_baseline"),
  titanic_baseline: gpuTorchSmokeCommand("titanic_baseline"),
  telco_churn_baseline: gpuTorchSmokeCommand("telco_churn_baseline"),
  all_tasks_baseline: gpuTorchSmokeCommand("all_tasks_baseline"),
  house_prices_seed_sweep: gpuTorchSmokeCommand("house_prices_seed_sweep"),
  titanic_seed_sweep: gpuTorchSmokeCommand("titanic_seed_sweep"),
  telco_churn_seed_sweep: gpuTorchSmokeCommand("telco_churn_seed_sweep"),
  all_tasks_seed_sweep: gpuTorchSmokeCommand("all_tasks_seed_sweep"),
  playground_s6e6_pytorch_mlp: "__WORKSTATION_LOCAL_S6E6_PYTORCH_MLP__",
  playground_s6e6_ensemble: "__WORKSTATION_LOCAL_S6E6_ENSEMBLE_LGB_XGB_CAT__",
  playground_s6e6_boosting_ensemble: "__WORKSTATION_LOCAL_S6E6_BOOSTING_ENSEMBLE__",
  playground_s6e6_lgbm_optuna: "__WORKSTATION_LOCAL_S6E6_LGBM_OPTUNA__",
  playground_s6e6_lightgbm: "__WORKSTATION_LOCAL_S6E6_LIGHTGBM__",
  playground_s6e6_xgboost: "__WORKSTATION_LOCAL_S6E6_XGBOOST__",
  playground_s6e6_catboost: "__WORKSTATION_LOCAL_S6E6_CATBOOST__"
} as const;

type GpuTemplate = keyof typeof allowedTemplates;

function missingEnv() {
  const config = gpuSshConfig();
  return [
    !config.host ? "GPU_SSH_HOST" : null,
    !config.username ? "GPU_SSH_USER" : null,
    !config.keyPath && !config.password ? "GPU_SSH_KEY_PATH_OR_GPU_SSH_PASSWORD" : null,
    !config.remoteWorkspace ? "GPU_REMOTE_WORKSPACE" : null
  ].filter(Boolean) as string[];
}

function workspacePolicyViolation(remoteWorkspace: string) {
  const normalized = remoteWorkspace.replace(/\/+$/, "");
  if (!normalized) return "GPU remote workspace must be configured explicitly.";
  if (!normalized.startsWith("/") && !normalized.startsWith("~/")) {
    return "GPU remote workspace must be an absolute POSIX path or start with ~/.";
  }
  if (/[\u0000-\u001f\u007f]/.test(normalized)) return "GPU remote workspace contains control characters.";
  const parts = normalized.split("/").filter(Boolean);
  if (parts.some((part) => part === "." || part === "..")) {
    return "GPU remote workspace must not contain traversal segments.";
  }
  const sharedRoots = new Set(["/", "/home", "/root", "/tmp", "/usr", "/var", "/opt", "~"]);
  if (sharedRoots.has(normalized)) {
    return "GPU remote workspace must be a dedicated project directory, not a shared root.";
  }
  return "";
}

async function rejectWorkspacePolicyViolation(action: string): Promise<GpuGatewayResult | null> {
  const config = gpuSshConfig();
  const violation = workspacePolicyViolation(config.remoteWorkspace);
  if (!violation) return null;
  const artifact = await writeJsonArtifact(`workspace/gpu/workspace_policy_blocked_${stamp()}.json`, {
    schema: "academic_research_os.gpu_workspace_policy_block.v1",
    action,
    status: "rejected",
    reason: violation,
    remote_workspace: config.remoteWorkspace,
    required_remote_workspace: "operator-configured dedicated POSIX project directory",
    training_started: false,
    policy: "Do not write workstation scripts, logs, datasets, or results directly under a shared remote root."
  });
  await logAction({
    action: "gpu_workspace_policy_blocked",
    message: violation,
    artifactPath: artifact,
    metadata: { remote_workspace: config.remoteWorkspace, required_remote_workspace: "dedicated_project_directory" }
  });
  return {
    ok: true,
    configured: hasGpuSshConfig(),
    status: "rejected",
    provider: "ssh_gateway",
    host: config.host,
    username: config.username,
    remote_workspace: config.remoteWorkspace,
    error: violation,
    artifact_path: artifact
  };
}

function gpuTorchSmokeCommand(label: string) {
  return `set -e
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
fi
if [ -z "$PYTHON_BIN" ]; then
  echo '{"template": ${JSON.stringify(label)}, "python_runtime": "missing", "cuda_visible_by_nvidia_smi": true}'
  exit 0
fi
$PYTHON_BIN - <<'PY'
import json

label = ${JSON.stringify(label)}
result = {
    "template": label,
    "python_runtime": "available",
    "cuda_visible_by_nvidia_smi": True,
}
try:
    import torch
    result.update({
        "torch_import": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    })
    if torch.cuda.is_available():
        x = torch.ones((512, 512), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        result["cuda_matmul_sum"] = float(y.sum().item())
        result["device0"] = torch.cuda.get_device_name(0)
except Exception as exc:
    result.update({
        "torch_import": False,
        "torch_error": str(exc)[:240],
        "cuda_available": None,
        "cuda_device_count": None,
    })
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
PY
`;
}

function sshBaseArgs() {
  const config = gpuSshConfig();
  const host = requireNetworkHost(config.host, "GPU SSH host");
  const port = requireTcpPort(config.port, "GPU SSH port");
  const username = requireSshUsername(config.username, "GPU SSH username");
  const hostKeyArgs = ["-o", "StrictHostKeyChecking=yes"];
  if (config.knownHostsPath) {
    hostKeyArgs.push("-o", `UserKnownHostsFile=${config.knownHostsPath}`);
  }
  const proxyArgs = config.socksProxy.host
    ? [
        "-o",
        `ProxyCommand=${proxyCommand(config.socksProxy.host, config.socksProxy.port, config.socksProxy.username)}`
      ]
    : [];
  return [
    "-i", config.keyPath,
    "-p", port,
    "-o", "BatchMode=yes",
    ...hostKeyArgs,
    ...proxyArgs,
    "-o", "ConnectTimeout=12",
    `${username}@${host}`
  ];
}

function pythonCommand() {
  if (process.env.WORKSTATION_PYTHON) return process.env.WORKSTATION_PYTHON;
  if (process.platform !== "win32") return "python3";
  const candidates = [
    "C:\\codex-python\\python.exe",
    path.join(os.homedir(), ".cache", "codex-runtimes", "codex-primary-runtime", "dependencies", "python", "python.exe")
  ];
  return candidates.find((candidate) => fs.existsSync(candidate)) ?? "python";
}

async function runPasswordSshCommand(command: string, timeout: number, maxBuffer: number) {
  const config = gpuSshConfig();
  const host = requireNetworkHost(config.host, "GPU SSH host");
  const port = requireTcpPort(config.port, "GPU SSH port");
  const username = requireSshUsername(config.username, "GPU SSH username");
  const script = path.join(workspaceRoot, "scripts", "run_hpc_ssh_command.py");
  const args = [
    script,
    "--host", host,
    "--port", port,
    "--user", username,
    "--command", command
  ];
  if (config.socksProxy.host) {
    args.push("--proxy-host", config.socksProxy.host, "--proxy-port", config.socksProxy.port);
  }
  return execFileAsync(pythonCommand(), args, {
    timeout,
    maxBuffer,
    env: { ...process.env, GPU_SSH_PASSWORD: config.password }
  });
}

async function refreshLocalHpcSocksBridge() {
  const config = gpuSshConfig();
  if (config.socksProxy.host !== "127.0.0.1" || config.socksProxy.port !== "7890") return;
  const manager = path.join(workspaceRoot, "scripts", "manage_hpc_proxy_bridge.ps1");
  if (!fs.existsSync(manager)) return;
  const baseArgs = ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", manager];
  const bridgeEnv = { ...process.env };
  delete bridgeEnv.GPU_SSH_SOCKS_HOST;
  delete bridgeEnv.GPU_SSH_SOCKS_PORT;
  delete bridgeEnv.GPU_SSH_SOCKS_USER;
  delete bridgeEnv.GPU_SSH_SOCKS_PASSWORD;
  delete bridgeEnv.GPU_SSH_SOCKS_PASSWORD_FILE;
  try {
    const status = await execFileAsync("powershell", [...baseArgs, "status"], { timeout: 15_000, maxBuffer: 1024 * 256, env: bridgeEnv });
    const statusPayload = JSON.parse(status.stdout || "{}") as { status?: string };
    if (statusPayload.status === "running") return;
    await execFileAsync("powershell", [...baseArgs, "start"], { timeout: 45_000, maxBuffer: 1024 * 512, env: bridgeEnv });
    await new Promise((resolve) => setTimeout(resolve, 2_000));
  } catch {
    // Let the SSH command produce the authoritative blocker artifact if the bridge still fails.
  }
}

async function runRemoteCommand(command: string, timeout: number, maxBuffer: number) {
  const config = gpuSshConfig();
  await refreshLocalHpcSocksBridge();
  if (config.password) {
    return runPasswordSshCommand(safeRemoteCommand(command), timeout, maxBuffer);
  }
  return execFileAsync("ssh", [...sshBaseArgs(), safeRemoteCommand(command)], { timeout, maxBuffer });
}

function proxyCommandScriptPath() {
  const source = path.join(workspaceRoot, "scripts", "hpc_socks_proxy.py");
  const targetDir = path.join(os.tmpdir(), "research_agent_workstation");
  const target = path.join(targetDir, "hpc_socks_proxy.py");
  try {
    fs.mkdirSync(targetDir, { recursive: true });
    const sourceStat = fs.statSync(source);
    const targetStat = fs.existsSync(target) ? fs.statSync(target) : null;
    if (!targetStat || sourceStat.mtimeMs > targetStat.mtimeMs || sourceStat.size !== targetStat.size) {
      fs.copyFileSync(source, target);
    }
    return target;
  } catch {
    return source;
  }
}

function proxyCommand(host: string, port: string, username: string) {
  const safeHost = requireNetworkHost(host, "SOCKS proxy host");
  const safePort = requireTcpPort(port, "SOCKS proxy port");
  const safeUsername = requireOptionalProxyUsername(username);
  const scriptPath = requireShellSafePath(proxyCommandScriptPath(), "SOCKS proxy script path");
  const userArg = safeUsername ? ` ${safeUsername}` : "";
  const python = requireShellSafePath(pythonCommand(), "Python command path");
  if (process.platform === "win32") {
    return `"${python}" "${scriptPath}" ${safeHost} ${safePort} %h %p${userArg}`;
  }
  return `${shellSingleQuote(python)} ${shellSingleQuote(scriptPath)} ${shellSingleQuote(safeHost)} ${shellSingleQuote(safePort)} %h %p${safeUsername ? ` ${shellSingleQuote(safeUsername)}` : ""}`;
}

function hostKeyPolicy() {
  return gpuSshConfig().knownHostsPath ? "custom_known_hosts_required" : "system_known_hosts_required";
}

function proxyPolicy() {
  const proxy = gpuSshConfig().socksProxy;
  return proxy.host ? { type: "socks5", host: proxy.host, port: proxy.port, auth: proxy.username ? "username_password" : "none" } : { type: "direct" };
}

function authPolicy() {
  return gpuSshConfig().password ? "password_env_dpapi" : "private_key_path";
}

function isResourceGatewayError(message: string) {
  return /AuthenticationException|Authentication failed|Password authentication|ConnectionRefusedError|WinError 10061|timed out|No route to host|proxy|SOCKS|SSH protocol banner|kex_exchange_identification|Connection closed by remote host|EOFError|transport shut down|saw EOF/i.test(message);
}

function resourceGatewayBlocker(message: string) {
  const usingProxy = Boolean(gpuSshConfig().socksProxy.host);
  if (/AuthenticationException|Authentication failed|Password authentication/i.test(message)) {
    return "GPU SSH authentication did not complete for the current allocation.";
  }
  if (usingProxy) return "GPU SSH proxy is not reachable from this workstation process.";
  if (/SSH protocol banner|kex_exchange_identification|Connection closed by remote host|EOFError/i.test(message)) {
    return "GPU SSH endpoint accepted TCP but closed before completing the SSH handshake.";
  }
  return "GPU SSH endpoint is unreachable from this workstation process.";
}

function resourceGatewayNextAction(message: string) {
  const usingProxy = Boolean(gpuSshConfig().socksProxy.host);
  if (/AuthenticationException|Authentication failed|Password authentication/i.test(message)) {
    return "Refresh the rotating GPU allocation account/password in DPAPI, then rerun /api/gpu/connections/test.";
  }
  if (usingProxy) return "Start the documented SOCKS proxy or refresh the rotating GPU allocation, then rerun /api/gpu/connections/test.";
  if (/SSH protocol banner|kex_exchange_identification|Connection closed by remote host|EOFError/i.test(message)) {
    return "Verify the rotating GPU allocation endpoint/account in the provider console or request a fresh allocation; the current host:port closes before SSH authentication.";
  }
  return "Verify host, port, account, and network reachability, then rerun /api/gpu/connections/test.";
}

function sanitizeGatewayDiagnostics(message: string) {
  const withoutCommand = message.replace(/^Command failed:[\s\S]*?(?=Traceback \(most recent call last\):|Authentication|Error:|$)/i, "");
  return withoutCommand
    .replace(/[A-Z]:\\[^\r\n\t"']+/g, "<local-path>")
    .replace(new RegExp(workspaceRoot.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g"), "<workspace>")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(-8)
    .join("\n")
    .slice(0, 1200);
}

function summarizeGpuError(message: string, gatewayBlocked: boolean) {
  if (gatewayBlocked) return resourceGatewayBlocker(message);
  const sanitized = sanitizeGatewayDiagnostics(message);
  return sanitized || "GPU SSH/CUDA smoke failed.";
}

function safeRemoteCommand(command: string) {
  const config = gpuSshConfig();
  return `cd ${shellSingleQuote(config.remoteWorkspace)} && ${command}`;
}

function parseJsonObjectFromOutput(output: string): Record<string, unknown> {
  const trimmed = output.trim();
  for (let index = trimmed.indexOf("{"); index >= 0; index = trimmed.indexOf("{", index + 1)) {
    try {
      const payload = JSON.parse(trimmed.slice(index)) as unknown;
      if (payload && typeof payload === "object" && !Array.isArray(payload)) return payload as Record<string, unknown>;
    } catch {
      // Keep scanning: remote shells may print banners before the JSON object.
    }
  }
  throw new Error("Remote command output did not contain a JSON object.");
}

function s6e6BoostingDependencyCommand() {
  const venvPython = shellSingleQuote(s6e6BoostingPythonPath());
  return `set -e
PYTHON_BIN=""
if [ -x ${venvPython} ]; then
  PYTHON_BIN=${venvPython}
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
fi
if [ -z "$PYTHON_BIN" ]; then
  echo '{"schema":"academic_research_os.hpc_boosting_dependency_probe.v1","status":"failed","python_runtime":"missing","required_packages":["lightgbm","xgboost","catboost"]}'
  exit 0
fi
$PYTHON_BIN - <<'PY'
import importlib
import json
import platform

required = ["lightgbm", "xgboost", "catboost"]
packages = {}
for name in required:
    try:
        module = importlib.import_module(name)
        packages[name] = {
            "available": True,
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:
        packages[name] = {
            "available": False,
            "error": str(exc)[:240],
        }
missing = [name for name, info in packages.items() if not info["available"]]
print(json.dumps({
    "schema": "academic_research_os.hpc_boosting_dependency_probe.v1",
    "status": "passed" if not missing else "blocked_dependency",
    "python_runtime": "available",
    "python_version": platform.python_version(),
    "required_packages": required,
    "packages": packages,
    "missing_packages": missing,
    "training_started": False,
    "policy": "S6E6 score-improvement training may start only after LightGBM, XGBoost, and CatBoost all import successfully.",
}, ensure_ascii=False, sort_keys=True))
PY
`;
}

function s6e6BoostingEnvironmentBootstrapCommand() {
  const venvRoot = s6e6BoostingPythonPath().replace(/\/bin\/python$/, "");
  const venvRootQuoted = shellSingleQuote(venvRoot);
  const pythonQuoted = shellSingleQuote(s6e6BoostingPythonPath());
  return `set -e
BOOTSTRAP_ROOT=${venvRootQuoted}
mkdir -p "$(dirname "$BOOTSTRAP_ROOT")"
PYTHON_BASE=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BASE=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BASE=python
fi
if [ -z "$PYTHON_BASE" ]; then
  echo '{"schema":"academic_research_os.hpc_boosting_environment_bootstrap.v1","status":"failed","training_started":false,"error":"python runtime missing"}'
  exit 0
fi
VIRTUALENV_BIN=""
if command -v virtualenv >/dev/null 2>&1; then
  VIRTUALENV_BIN="$(command -v virtualenv)"
elif [ -x "$HOME/.local/bin/virtualenv" ]; then
  VIRTUALENV_BIN="$HOME/.local/bin/virtualenv"
fi
if [ ! -x ${pythonQuoted} ] || ! ${pythonQuoted} -m pip --version >/dev/null 2>&1; then
  if [ -n "$VIRTUALENV_BIN" ]; then
    "$VIRTUALENV_BIN" --clear --python "$PYTHON_BASE" "$BOOTSTRAP_ROOT"
  else
    "$PYTHON_BASE" -m venv --clear "$BOOTSTRAP_ROOT"
  fi
fi
if ! ${pythonQuoted} -m pip --version >/dev/null 2>&1; then
  GET_PIP="$BOOTSTRAP_ROOT/get-pip.py"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$GET_PIP"
  elif command -v wget >/dev/null 2>&1; then
    wget -q https://bootstrap.pypa.io/get-pip.py -O "$GET_PIP"
  else
    echo '{"schema":"academic_research_os.hpc_boosting_environment_bootstrap.v1","status":"failed","training_started":false,"error":"pip bootstrap unavailable: no virtualenv pip, curl, or wget"}'
    exit 0
  fi
  ${pythonQuoted} "$GET_PIP"
fi
if ! ${pythonQuoted} -m pip --version >/dev/null 2>&1; then
  echo '{"schema":"academic_research_os.hpc_boosting_environment_bootstrap.v1","status":"failed","training_started":false,"error":"pip bootstrap failed"}'
  exit 0
fi
${pythonQuoted} -m pip install --upgrade pip setuptools wheel
${pythonQuoted} -m pip install --upgrade numpy pandas scipy scikit-learn lightgbm xgboost catboost optuna
${pythonQuoted} - <<'PY'
import importlib
import json
import platform
import sys

required = ["numpy", "pandas", "sklearn", "scipy", "lightgbm", "xgboost", "catboost", "optuna"]
packages = {}
for name in required:
    try:
        module = importlib.import_module(name)
        packages[name] = {
            "available": True,
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:
        packages[name] = {
            "available": False,
            "error": str(exc)[:240],
        }
missing = [name for name, info in packages.items() if not info["available"]]
print(json.dumps({
    "schema": "academic_research_os.hpc_boosting_environment_bootstrap.v1",
    "status": "passed" if not missing else "blocked_dependency",
    "python_runtime": "available",
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "required_packages": required,
    "packages": packages,
    "missing_packages": missing,
    "training_started": False,
    "policy": "Environment bootstrap may install dependencies only; training and Kaggle submit remain behind workstation gates.",
}, ensure_ascii=False, sort_keys=True))
PY
`;
}

export async function bootstrapS6E6BoostingEnvironment(): Promise<GpuGatewayResult> {
  const config = gpuSshConfig();
  if (!hasGpuSshConfig()) {
    const artifact = await writeJsonArtifact(`workspace/gpu/s6e6_boosting_environment_bootstrap_${stamp()}.json`, {
      schema: "academic_research_os.hpc_boosting_environment_bootstrap.v1",
      provider: "ssh_gateway",
      status: "not_configured",
      configured: false,
      missing_env: missingEnv(),
      training_started: false,
      created_at: new Date().toISOString()
    });
    return { ok: true, configured: false, status: "not_configured", provider: "ssh_gateway", missing_env: missingEnv(), artifact_path: artifact };
  }
  try {
    const { stdout, stderr } = await runRemoteCommand(s6e6BoostingEnvironmentBootstrapCommand(), 1000 * 60 * 20, 1024 * 1024 * 12);
    const payload = parseJsonObjectFromOutput(stdout);
    const status = payload.status === "passed" ? "passed" : payload.status === "blocked_dependency" ? "blocked_dependency" : "failed";
    const artifact = await writeJsonArtifact(`workspace/gpu/s6e6_boosting_environment_bootstrap_${stamp()}.json`, {
      ...payload,
      provider: "ssh_gateway",
      configured: true,
      host: config.host,
      username: config.username,
      remote_workspace: config.remoteWorkspace,
      remote_python: s6e6BoostingPythonPath(),
      host_key_policy: hostKeyPolicy(),
      proxy_policy: proxyPolicy(),
      auth_policy: authPolicy(),
      stdout_tail: stdout.slice(-4000),
      stderr_tail: stderr.slice(-4000),
      created_at: new Date().toISOString()
    });
    await logAction({
      action: status === "passed" ? "s6e6_boosting_environment_bootstrap_passed" : "s6e6_boosting_environment_bootstrap_blocked",
      message: status === "passed"
        ? "S6E6 boosting Python environment is ready on remote HPC."
        : "S6E6 boosting Python environment bootstrap did not satisfy dependency gate.",
      artifactPath: artifact,
      metadata: { host: config.host, status, remote_python: s6e6BoostingPythonPath(), missing_packages: payload.missing_packages ?? [] }
    });
    return { ok: true, configured: true, status, provider: "ssh_gateway", host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, stdout, stderr, artifact_path: artifact };
  } catch (error) {
    const message = error instanceof Error ? error.message : "S6E6 boosting environment bootstrap failed.";
    const artifact = await writeJsonArtifact(`workspace/gpu/s6e6_boosting_environment_bootstrap_${stamp()}.json`, {
      schema: "academic_research_os.hpc_boosting_environment_bootstrap.v1",
      provider: "ssh_gateway",
      configured: true,
      status: "failed",
      host: config.host,
      username: config.username,
      remote_workspace: config.remoteWorkspace,
      remote_python: s6e6BoostingPythonPath(),
      training_started: false,
      error: message,
      created_at: new Date().toISOString()
    });
    await logAction({ action: "s6e6_boosting_environment_bootstrap_failed", message, artifactPath: artifact, metadata: { host: config.host, remote_python: s6e6BoostingPythonPath() } });
    return { ok: true, configured: true, status: "failed", provider: "ssh_gateway", host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, error: message, artifact_path: artifact };
  }
}

export async function testS6E6BoostingDependencies(): Promise<GpuGatewayResult> {
  const config = gpuSshConfig();
  if (!hasGpuSshConfig()) {
    const artifact = await writeJsonArtifact(`workspace/gpu/s6e6_boosting_dependency_${stamp()}.json`, {
      schema: "academic_research_os.hpc_boosting_dependency_probe.v1",
      provider: "ssh_gateway",
      status: "not_configured",
      configured: false,
      missing_env: missingEnv(),
      training_started: false,
      created_at: new Date().toISOString()
    });
    return { ok: true, configured: false, status: "not_configured", provider: "ssh_gateway", missing_env: missingEnv(), artifact_path: artifact };
  }
  try {
    const { stdout, stderr } = await runRemoteCommand(s6e6BoostingDependencyCommand(), 1000 * 45, 1024 * 1024 * 2);
    const payload = parseJsonObjectFromOutput(stdout);
    const status = payload.status === "passed" ? "passed" : payload.status === "blocked_dependency" ? "blocked_dependency" : "failed";
    const artifact = await writeJsonArtifact(`workspace/gpu/s6e6_boosting_dependency_${stamp()}.json`, {
      ...payload,
      provider: "ssh_gateway",
      configured: true,
      host: config.host,
      username: config.username,
      remote_workspace: config.remoteWorkspace,
      host_key_policy: hostKeyPolicy(),
      proxy_policy: proxyPolicy(),
      auth_policy: authPolicy(),
      stdout_tail: stdout.slice(-4000),
      stderr_tail: stderr.slice(-2000),
      created_at: new Date().toISOString()
    });
    await logAction({
      action: status === "passed" ? "s6e6_boosting_dependency_gate_passed" : "s6e6_boosting_dependency_gate_blocked",
      message: status === "passed"
        ? "S6E6 boosting dependency gate passed on the remote HPC runtime."
        : "S6E6 boosting dependency gate blocked long training before remote dependencies were ready.",
      artifactPath: artifact,
      metadata: { host: config.host, status, missing_packages: payload.missing_packages ?? [] }
    });
    return { ok: true, configured: true, status, provider: "ssh_gateway", host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, stdout, stderr, artifact_path: artifact };
  } catch (error) {
    const message = error instanceof Error ? error.message : "S6E6 boosting dependency probe failed.";
    const gatewayBlocked = isResourceGatewayError(message);
    const status = gatewayBlocked ? "blocked_resource_gateway" : "failed";
    const artifact = await writeJsonArtifact(`workspace/gpu/s6e6_boosting_dependency_${stamp()}.json`, {
      schema: "academic_research_os.hpc_boosting_dependency_probe.v1",
      provider: "ssh_gateway",
      configured: true,
      status,
      host: config.host,
      username: config.username,
      remote_workspace: config.remoteWorkspace,
      training_started: false,
      error: message,
      blocker: gatewayBlocked ? resourceGatewayBlocker(message) : "GPU dependency probe failed before training.",
      next_action: gatewayBlocked ? resourceGatewayNextAction(message) : "Review the dependency probe artifact and rerun after the remote runtime is repaired.",
      created_at: new Date().toISOString()
    });
    await logAction({
      action: gatewayBlocked ? "s6e6_boosting_dependency_gate_resource_blocked" : "s6e6_boosting_dependency_gate_failed",
      message: gatewayBlocked ? "S6E6 boosting dependency gate blocked because the GPU SSH resource gateway is unreachable." : message,
      artifactPath: artifact,
      metadata: { host: config.host, status, resource_gateway_blocked: gatewayBlocked }
    });
    return { ok: true, configured: true, status, provider: "ssh_gateway", host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, error: message, artifact_path: artifact };
  }
}

export async function testGpuConnection(): Promise<GpuGatewayResult> {
  const config = gpuSshConfig();
  if (!hasGpuSshConfig()) {
    const artifact = await writeJsonArtifact(`workspace/gpu/connection_test_${stamp()}.json`, {
      provider: "ssh_gateway",
      status: "not_configured",
      missing_env: missingEnv(),
      created_at: new Date().toISOString()
    });
    await logAction({
      action: "gpu_connection_not_configured",
      message: "GPU SSH gateway is not configured. Set GPU_SSH_HOST, GPU_SSH_USER, GPU_SSH_PASSWORD or GPU_SSH_KEY_PATH, and GPU_REMOTE_WORKSPACE.",
      artifactPath: artifact,
      metadata: { provider: "ssh_gateway", status: gpuSshStatus(), missing_env: missingEnv() }
    });
    return { ok: true, configured: false, status: "not_configured", provider: "ssh_gateway", missing_env: missingEnv(), artifact_path: artifact };
  }
  const workspacePolicyBlock = await rejectWorkspacePolicyViolation("test_gpu_connection");
  if (workspacePolicyBlock) return workspacePolicyBlock;

  try {
    const gpuSmoke = `set -e
pwd
test -w . && echo GPU_WORKSPACE_WRITABLE
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
fi
if [ -z "$PYTHON_BIN" ]; then
  echo '{"python_runtime":"missing","torch_import":false,"cuda_visible_by_nvidia_smi":true}'
  exit 0
fi
$PYTHON_BIN - <<'PY'
import json

payload = {
    "python_runtime": "available",
    "cuda_visible_by_nvidia_smi": True,
}
try:
    import torch
    payload.update({
        "torch_import": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    })
    if torch.cuda.is_available():
        x = torch.ones((256, 256), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        payload["cuda_matmul_sum"] = float(y.sum().item())
except Exception as exc:
    payload.update({
        "torch_import": False,
        "torch_error": str(exc)[:240],
        "cuda_available": None,
        "cuda_device_count": None,
    })
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
PY`;
    const { stdout, stderr } = await runRemoteCommand(gpuSmoke, 30000, 1024 * 1024);
    const artifact = await writeJsonArtifact(`workspace/gpu/connection_test_${stamp()}.json`, {
      provider: "ssh_gateway",
      status: "passed",
      host: config.host,
      username: config.username,
      remote_workspace: config.remoteWorkspace,
      host_key_policy: hostKeyPolicy(),
      proxy_policy: proxyPolicy(),
      auth_policy: authPolicy(),
      stdout,
      stderr,
      created_at: new Date().toISOString()
    });
    await logAction({ action: "gpu_connection_test_passed", message: "GPU SSH connection test passed.", artifactPath: artifact, metadata: { host: config.host, remote_workspace: config.remoteWorkspace } });
    return { ok: true, configured: true, status: "passed", provider: "ssh_gateway", host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, stdout, stderr, artifact_path: artifact };
  } catch (error) {
    const message = error instanceof Error ? error.message : "GPU SSH connection test failed.";
    const gatewayBlocked = isResourceGatewayError(message);
    const status = gatewayBlocked ? "blocked_resource_gateway" : "failed";
    const errorSummary = summarizeGpuError(message, gatewayBlocked);
    const diagnosticsTail = sanitizeGatewayDiagnostics(message);
    const artifact = await writeJsonArtifact(`workspace/gpu/connection_test_${stamp()}.json`, {
      provider: "ssh_gateway",
      status,
      host: config.host,
      username: config.username,
      remote_workspace: config.remoteWorkspace,
      host_key_policy: hostKeyPolicy(),
      proxy_policy: proxyPolicy(),
      auth_policy: authPolicy(),
      blocker: gatewayBlocked ? resourceGatewayBlocker(message) : "GPU SSH/CUDA smoke failed.",
      next_action: gatewayBlocked ? resourceGatewayNextAction(message) : "Review the smoke artifact and rerun after the remote runtime is repaired.",
      training_started: false,
      error: errorSummary,
      diagnostics_tail: diagnosticsTail,
      created_at: new Date().toISOString()
    });
    await logAction({
      action: gatewayBlocked ? "gpu_current_allocation_blocker" : "gpu_connection_test_failed",
      message: errorSummary,
      artifactPath: artifact,
      metadata: {
        status: gatewayBlocked ? "blocked_current_allocation" : status,
        host: config.host,
        port: config.port,
        user: config.username,
        remote_workspace: config.remoteWorkspace,
        proxy: config.socksProxy.host ? `${config.socksProxy.host}:${config.socksProxy.port}` : "direct",
        tcp_direct: config.socksProxy.host ? "not_tested_proxy_mode" : "passed",
        ssh_direct: /SSH protocol banner|kex_exchange_identification|Connection closed by remote host|EOFError/i.test(message) ? "closed_before_kex" : "failed_before_auth",
        training_started: false,
        official_submission_started: false,
        next_action: gatewayBlocked ? resourceGatewayNextAction(message) : "Review the GPU connection artifact."
      }
    });
    return { ok: true, configured: true, status, provider: "ssh_gateway", host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, error: errorSummary, artifact_path: artifact };
  }
}

function isGpuTemplate(value: string): value is GpuTemplate {
  return Object.prototype.hasOwnProperty.call(allowedTemplates, value);
}

function defaultTemplateForTask(taskId: string): GpuTemplate {
  if (taskId === "playground_series_s6e6") return "playground_s6e6_boosting_ensemble";
  if (taskId === "titanic") return "titanic_baseline";
  if (taskId === "telco_churn") return "telco_churn_baseline";
  return "house_prices_baseline";
}

function relativeFromRoot(targetPath: string) {
  return path.relative(workspaceRoot, targetPath).replaceAll("\\", "/");
}

async function fileExists(relativePath: string) {
  return fs.promises.stat(path.join(workspaceRoot, relativePath)).then((stat) => stat.isFile()).catch(() => false);
}

async function writeFreshGpuSmokeBlocker(input: {
  jobId: string;
  taskId: string;
  runId: string | null;
  agentId: string;
  gateId: string | null;
  template: string;
  connection: GpuGatewayResult;
}) {
  const config = gpuSshConfig();
  const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}.json`, {
    schema: "academic_research_os.gpu_job_result.v1",
    job_id: input.jobId,
    task_id: input.taskId,
    workstation_run_id: input.runId,
    agent_id: input.agentId,
    gate_id: input.gateId,
    template: input.template,
    provider: "ssh_gateway",
    status: "blocked_resource_gateway",
    reason: "Fresh GPU SSH/CUDA smoke must pass before any non-smoke GPU/HPC training job starts.",
    connection_status: input.connection.status,
    connection_artifact: input.connection.artifact_path ?? null,
    remote_workspace: config.remoteWorkspace,
    host_key_policy: hostKeyPolicy(),
    proxy_policy: proxyPolicy(),
    auth_policy: authPolicy(),
    training_started: false,
    official_submission_started: false,
    retry_policy: "Refresh the rotating GPU allocation or proxy credentials, rerun /api/gpu/connections/test, then resubmit through the workstation gate.",
    created_at: new Date().toISOString()
  });
  await logAction({
    action: "gpu_job_fresh_smoke_blocked",
    taskId: input.taskId,
    runId: input.runId ?? undefined,
    message: `GPU job ${input.template} blocked because fresh SSH/CUDA smoke did not pass.`,
    artifactPath: artifact,
    metadata: {
      job_id: input.jobId,
      template: input.template,
      connection_status: input.connection.status,
      connection_artifact: input.connection.artifact_path ?? null
    }
  });
  return {
    ok: true as const,
    configured: input.connection.configured,
    status: "blocked_resource_gateway" as const,
    provider: "ssh_gateway" as const,
    host: config.host,
    username: config.username,
    remote_workspace: config.remoteWorkspace,
    error: "Fresh GPU SSH/CUDA smoke did not pass; training was not started.",
    artifact_path: artifact,
    stderr_artifact: input.connection.artifact_path,
    pulled_artifacts: input.connection.artifact_path ? [input.connection.artifact_path] : []
  };
}

async function submitPlaygroundS6E6GpuJob(input: {
  jobId: string;
  taskId: string;
  runId: string | null;
  agentId: string;
  gateId: string | null;
  resourceRequest?: Record<string, unknown>;
}): Promise<GpuGatewayResult> {
  const config = gpuSshConfig();
  const localArtifactRoot = path.join(
    workspaceRoot,
    "workspace",
    "workstation_runs",
    input.taskId,
    input.runId ?? input.jobId,
    "hpc_gpu_training"
  );
  const script = path.join(workspaceRoot, "scripts", "run_hpc_kaggle_pytorch.py");
  const args = [
    script,
    "--host", config.host,
    "--port", config.port,
    "--user", config.username,
    "--password-env", "GPU_SSH_PASSWORD",
    "--remote-root", config.remoteWorkspace,
    "--local-artifact-dir", localArtifactRoot
  ];
  if (config.socksProxy.host) {
    args.push("--proxy-host", config.socksProxy.host, "--proxy-port", config.socksProxy.port);
  } else {
    args.push("--proxy-host", "");
  }

  const scriptArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.launcher.txt`, `${pythonCommand()} ${args.map((arg) => JSON.stringify(arg)).join(" ")}`);
  const preflightManifestPath = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}_manifest.json`, {
    schema: "academic_research_os.gpu_job_manifest.v1",
    job_id: input.jobId,
    task_id: input.taskId,
    workstation_run_id: input.runId,
    agent_id: input.agentId,
    gate_id: input.gateId,
    provider: "ssh_gateway",
    command_template: "playground_s6e6_pytorch_mlp",
    resource_request: input.resourceRequest ?? { gpu: "any_available", task: "playground_series_s6e6", mode: "full_data_training" },
    remote_workspace: config.remoteWorkspace,
    remote_python: s6e6BoostingPythonPath(),
    local_artifact_root: relativeFromRoot(localArtifactRoot),
    log_path: `workspace/workstation_runs/${input.taskId}/${input.runId ?? input.jobId}/hpc_gpu_training`,
    pullback_policy: "metrics_submission_report_stdout_stderr",
    timeout_seconds: 5400,
    cancel_record_path: `workspace/gpu/jobs/${input.jobId}_cancel.json`,
    status: "prepared",
    created_at: new Date().toISOString()
  });

  try {
    const { stdout, stderr } = await execFileAsync(pythonCommand(), args, {
      cwd: workspaceRoot,
      timeout: 1000 * 60 * 95,
      maxBuffer: 1024 * 1024 * 16,
      env: { ...process.env, GPU_SSH_PASSWORD: config.password }
    });
    const stdoutArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.stdout.log`, stdout);
    const stderrArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.stderr.log`, stderr);
    const localRootRelative = relativeFromRoot(localArtifactRoot);
    const manifestArtifact = `${localRootRelative}/manifest.json`;
    const metricsArtifact = `${localRootRelative}/metrics.json`;
    const submissionArtifact = `${localRootRelative}/submission.csv`;
    const reportArtifact = `${localRootRelative}/report.md`;
    const pulledArtifacts = [stdoutArtifact, stderrArtifact];
    for (const candidate of [manifestArtifact, metricsArtifact, submissionArtifact, reportArtifact]) {
      if (await fileExists(candidate)) pulledArtifacts.push(candidate);
    }
    const artifacts = [
      await artifactDescriptor(scriptArtifact, {
        artifact_type: "gpu_job_launcher",
        created_by_agent: input.agentId,
        stage: "hpc_execution",
        claim_binding: "S6E6 GPU job was launched through POST /api/gpu/jobs and a workstation whitelist template.",
        gate_dependency: "hpc_execution_approval"
      }),
      await artifactDescriptor(preflightManifestPath, {
        artifact_type: "gpu_job_manifest",
        created_by_agent: input.agentId,
        stage: "hpc_execution",
        claim_binding: "The GPU execution has task, run, agent, gate, resource and pullback fields.",
        gate_dependency: "hpc_execution_approval"
      }),
      await artifactDescriptor(stdoutArtifact, {
        artifact_type: "remote_stdout",
        created_by_agent: input.agentId,
        stage: "hpc_execution",
        claim_binding: "Remote stdout was archived.",
        gate_dependency: "hpc_execution_approval"
      }),
      await artifactDescriptor(stderrArtifact, {
        artifact_type: "remote_stderr",
        created_by_agent: input.agentId,
        stage: "hpc_execution",
        claim_binding: "Remote stderr was archived.",
        gate_dependency: "hpc_execution_approval"
      })
    ];
    for (const [artifactPath, artifactType, claim] of [
      [metricsArtifact, "metrics", "Remote training metrics were pulled back."],
      [submissionArtifact, "submission", "Remote training produced a Kaggle submission file."],
      [reportArtifact, "hpc_report", "Remote training report was pulled back."]
    ] as const) {
      if (await fileExists(artifactPath)) {
        artifacts.push(await artifactDescriptor(artifactPath, {
          artifact_type: artifactType,
          created_by_agent: input.agentId,
          stage: "hpc_execution",
          claim_binding: claim,
          gate_dependency: "hpc_execution_approval"
        }));
      }
    }
    const artifactManifestPath = await writeArtifactManifestArtifact({
      taskId: input.taskId,
      runId: input.runId ?? input.jobId,
      relativePath: `workspace/gpu/jobs/${input.jobId}_artifact_manifest.json`,
      artifacts,
      source: "gpu_ssh_gateway",
      extra: { job_id: input.jobId, pulled_artifacts: pulledArtifacts }
    });
    const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}.json`, {
      schema: "academic_research_os.gpu_job_result.v1",
      job_id: input.jobId,
      task_id: input.taskId,
      workstation_run_id: input.runId,
      agent_id: input.agentId,
      gate_id: input.gateId,
      template: "playground_s6e6_pytorch_mlp",
      provider: "ssh_gateway",
      status: "submitted",
      command_template: "playground_s6e6_pytorch_mlp",
      remote_workspace: config.remoteWorkspace,
      local_artifact_root: localRootRelative,
      host_key_policy: hostKeyPolicy(),
      proxy_policy: proxyPolicy(),
      auth_policy: authPolicy(),
      script_artifact: scriptArtifact,
      job_manifest_path: preflightManifestPath,
      stdout_artifact: stdoutArtifact,
      stderr_artifact: stderrArtifact,
      metrics_artifact: await fileExists(metricsArtifact) ? metricsArtifact : null,
      submission_artifact: await fileExists(submissionArtifact) ? submissionArtifact : null,
      report_artifact: await fileExists(reportArtifact) ? reportArtifact : null,
      artifact_manifest_path: artifactManifestPath,
      pulled_artifacts: pulledArtifacts,
      stdout,
      stderr,
      created_at: new Date().toISOString()
    });
    await logAction({
      action: "gpu_job_submitted",
      taskId: input.taskId,
      runId: input.runId ?? undefined,
      message: "S6E6 GPU training completed through the workstation whitelist template.",
      artifactPath: artifact,
      metadata: { job_id: input.jobId, template: "playground_s6e6_pytorch_mlp", job_manifest_path: preflightManifestPath, artifact_manifest_path: artifactManifestPath }
    });
    return {
      ok: true,
      configured: true,
      status: "submitted",
      provider: "ssh_gateway",
      host: config.host,
      username: config.username,
      remote_workspace: config.remoteWorkspace,
      stdout,
      stderr,
      artifact_path: artifact,
      job_manifest_path: preflightManifestPath,
      stdout_artifact: stdoutArtifact,
      stderr_artifact: stderrArtifact,
      metrics_artifact: await fileExists(metricsArtifact) ? metricsArtifact : undefined,
      submission_artifact: await fileExists(submissionArtifact) ? submissionArtifact : undefined,
      report_artifact: await fileExists(reportArtifact) ? reportArtifact : undefined,
      artifact_manifest_path: artifactManifestPath,
      pulled_artifacts: pulledArtifacts
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "S6E6 GPU job failed.";
    const errorArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.error.log`, message);
    const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}.json`, {
      schema: "academic_research_os.gpu_job_result.v1",
      job_id: input.jobId,
      task_id: input.taskId,
      workstation_run_id: input.runId,
      agent_id: input.agentId,
      gate_id: input.gateId,
      template: "playground_s6e6_pytorch_mlp",
      provider: "ssh_gateway",
      status: "failed",
      remote_workspace: config.remoteWorkspace,
      host_key_policy: hostKeyPolicy(),
      proxy_policy: proxyPolicy(),
      auth_policy: authPolicy(),
      script_artifact: scriptArtifact,
      job_manifest_path: preflightManifestPath,
      stderr_artifact: errorArtifact,
      error: message,
      created_at: new Date().toISOString()
    });
    await logAction({
      action: "gpu_job_failed",
      taskId: input.taskId,
      runId: input.runId ?? undefined,
      message,
      artifactPath: artifact,
      metadata: { job_id: input.jobId, template: "playground_s6e6_pytorch_mlp", job_manifest_path: preflightManifestPath }
    });
    return {
      ok: true,
      configured: true,
      status: "failed",
      provider: "ssh_gateway",
      host: config.host,
      username: config.username,
      remote_workspace: config.remoteWorkspace,
      error: message,
      artifact_path: artifact,
      job_manifest_path: preflightManifestPath,
      stderr_artifact: errorArtifact,
      pulled_artifacts: [errorArtifact]
    };
  }
}

async function submitPlaygroundS6E6EnsembleGpuJob(input: {
  jobId: string;
  taskId: string;
  runId: string | null;
  agentId: string;
  gateId: string | null;
  resourceRequest?: Record<string, unknown>;
}): Promise<GpuGatewayResult> {
  const config = gpuSshConfig();
  const localArtifactRoot = path.join(
    workspaceRoot,
    "workspace",
    "workstation_runs",
    input.taskId,
    input.runId ?? input.jobId,
    "hpc_gpu_training"
  );
  const script = path.join(workspaceRoot, "scripts", "run_hpc_kaggle_ensemble.py");
  const args = [
    script,
    "--host", config.host,
    "--port", config.port,
    "--user", config.username,
    "--password-env", "GPU_SSH_PASSWORD",
    "--remote-root", config.remoteWorkspace,
    "--local-artifact-dir", localArtifactRoot,
    "--remote-python", s6e6BoostingPythonPath(),
    "--timeout-seconds", "10800"
  ];
  if (config.socksProxy.host) {
    args.push("--proxy-host", config.socksProxy.host, "--proxy-port", config.socksProxy.port);
  } else {
    args.push("--proxy-host", "");
  }

  const scriptArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.launcher.txt`, `${pythonCommand()} ${args.map((arg) => JSON.stringify(arg)).join(" ")}`);
  const preflightManifestPath = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}_manifest.json`, {
    schema: "academic_research_os.gpu_job_manifest.v1",
    job_id: input.jobId, task_id: input.taskId, workstation_run_id: input.runId,
    agent_id: input.agentId, gate_id: input.gateId, provider: "ssh_gateway",
    command_template: "playground_s6e6_ensemble",
    resource_request: input.resourceRequest ?? { gpu: "any_available", task: "playground_series_s6e6", mode: "ensemble_lgb_xgb_cat", n_folds: 5, n_seeds: 3 },
    remote_workspace: config.remoteWorkspace,
    local_artifact_root: relativeFromRoot(localArtifactRoot),
    log_path: `workspace/workstation_runs/${input.taskId}/${input.runId ?? input.jobId}/hpc_gpu_training`,
    pullback_policy: "metrics_submission_report_stdout_stderr",
    timeout_seconds: 7200,
    cancel_record_path: `workspace/gpu/jobs/${input.jobId}_cancel.json`,
    status: "prepared",
    created_at: new Date().toISOString()
  });

  try {
    const { stdout, stderr } = await execFileAsync(pythonCommand(), args, {
      cwd: workspaceRoot, timeout: 1000 * 60 * 125, maxBuffer: 1024 * 1024 * 16,
      env: { ...process.env, GPU_SSH_PASSWORD: config.password }
    });
    const stdoutArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.stdout.log`, stdout);
    const stderrArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.stderr.log`, stderr);
    const localRootRelative = relativeFromRoot(localArtifactRoot);
    const manifestArtifact = `${localRootRelative}/manifest.json`;
    const metricsArtifact = `${localRootRelative}/metrics.json`;
    const submissionArtifact = `${localRootRelative}/submission.csv`;
    const reportArtifact = `${localRootRelative}/report.md`;
    const pulledArtifacts = [stdoutArtifact, stderrArtifact];
    for (const candidate of [manifestArtifact, metricsArtifact, submissionArtifact, reportArtifact]) {
      if (await fileExists(candidate)) pulledArtifacts.push(candidate);
    }
    const artifacts = [
      await artifactDescriptor(scriptArtifact, {
        artifact_type: "gpu_job_launcher", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: "S6E6 ensemble GPU job launched through workstation whitelist template.",
        gate_dependency: "hpc_execution_approval"
      }),
      await artifactDescriptor(preflightManifestPath, {
        artifact_type: "gpu_job_manifest", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: "GPU ensemble job manifest includes task, run, agent, gate, resource and pullback.",
        gate_dependency: "hpc_execution_approval"
      }),
      await artifactDescriptor(stdoutArtifact, {
        artifact_type: "remote_stdout", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: "Ensemble remote stdout was archived.", gate_dependency: "hpc_execution_approval"
      }),
      await artifactDescriptor(stderrArtifact, {
        artifact_type: "remote_stderr", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: "Ensemble remote stderr was archived.", gate_dependency: "hpc_execution_approval"
      })
    ];
    for (const [artifactPath, artifactType, claim] of [
      [metricsArtifact, "ensemble_metrics", "Remote ensemble training metrics were pulled back."],
      [submissionArtifact, "ensemble_submission", "Remote ensemble training produced a Kaggle submission file."],
      [reportArtifact, "ensemble_hpc_report", "Remote ensemble training report was pulled back."]
    ] as const) {
      if (await fileExists(artifactPath)) {
        artifacts.push(await artifactDescriptor(artifactPath, {
          artifact_type: artifactType, created_by_agent: input.agentId, stage: "hpc_execution",
          claim_binding: claim, gate_dependency: "hpc_execution_approval"
        }));
      }
    }
    const artifactManifestPath = await writeArtifactManifestArtifact({
      taskId: input.taskId, runId: input.runId ?? input.jobId,
      relativePath: `workspace/gpu/jobs/${input.jobId}_artifact_manifest.json`,
      artifacts, source: "gpu_ssh_gateway_ensemble", extra: { job_id: input.jobId, pulled_artifacts: pulledArtifacts }
    });
    const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}.json`, {
      schema: "academic_research_os.gpu_job_result.v1",
      job_id: input.jobId, task_id: input.taskId, workstation_run_id: input.runId,
      agent_id: input.agentId, gate_id: input.gateId, template: "playground_s6e6_ensemble",
      provider: "ssh_gateway", status: "submitted", command_template: "playground_s6e6_ensemble",
      remote_workspace: config.remoteWorkspace, local_artifact_root: localRootRelative,
      host_key_policy: hostKeyPolicy(), proxy_policy: proxyPolicy(), auth_policy: authPolicy(),
      script_artifact: scriptArtifact, job_manifest_path: preflightManifestPath,
      stdout_artifact: stdoutArtifact, stderr_artifact: stderrArtifact,
      metrics_artifact: await fileExists(metricsArtifact) ? metricsArtifact : null,
      submission_artifact: await fileExists(submissionArtifact) ? submissionArtifact : null,
      report_artifact: await fileExists(reportArtifact) ? reportArtifact : null,
      artifact_manifest_path: artifactManifestPath, pulled_artifacts: pulledArtifacts,
      stdout, stderr, created_at: new Date().toISOString()
    });
    await logAction({ action: "gpu_ensemble_job_submitted", taskId: input.taskId, runId: input.runId ?? undefined, message: "S6E6 ensemble GPU training completed.", artifactPath: artifact, metadata: { job_id: input.jobId, template: "playground_s6e6_ensemble" } });
    return {
      ok: true, configured: true, status: "submitted", provider: "ssh_gateway",
      host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, stdout, stderr,
      artifact_path: artifact, job_manifest_path: preflightManifestPath, stdout_artifact: stdoutArtifact,
      stderr_artifact: stderrArtifact,
      metrics_artifact: await fileExists(metricsArtifact) ? metricsArtifact : undefined,
      submission_artifact: await fileExists(submissionArtifact) ? submissionArtifact : undefined,
      report_artifact: await fileExists(reportArtifact) ? reportArtifact : undefined,
      artifact_manifest_path: artifactManifestPath, pulled_artifacts: pulledArtifacts
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "S6E6 ensemble GPU job failed.";
    const errorArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.error.log`, message);
    const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}.json`, {
      schema: "academic_research_os.gpu_job_result.v1", job_id: input.jobId, task_id: input.taskId,
      workstation_run_id: input.runId, agent_id: input.agentId, gate_id: input.gateId,
      template: "playground_s6e6_ensemble", provider: "ssh_gateway", status: "failed",
      remote_workspace: config.remoteWorkspace, host_key_policy: hostKeyPolicy(),
      proxy_policy: proxyPolicy(), auth_policy: authPolicy(), script_artifact: scriptArtifact,
      job_manifest_path: preflightManifestPath, stderr_artifact: errorArtifact, error: message,
      created_at: new Date().toISOString()
    });
    await logAction({ action: "gpu_ensemble_job_failed", taskId: input.taskId, runId: input.runId ?? undefined, message, artifactPath: artifact, metadata: { job_id: input.jobId, template: "playground_s6e6_ensemble" } });
    return { ok: true, configured: true, status: "failed", provider: "ssh_gateway", host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, error: message, artifact_path: artifact, job_manifest_path: preflightManifestPath, stderr_artifact: errorArtifact, pulled_artifacts: [errorArtifact] };
  }
}

async function submitPlaygroundS6E6BoostingEnsembleGpuJob(input: {
  jobId: string;
  taskId: string;
  runId: string | null;
  agentId: string;
  gateId: string | null;
  template: string;
  resourceRequest?: Record<string, unknown>;
}): Promise<GpuGatewayResult> {
  const config = gpuSshConfig();
  const folds = integerResource(input.resourceRequest?.folds ?? input.resourceRequest?.n_folds, 5, 2, 5);
  const sampleRows = integerResource(input.resourceRequest?.sample_rows, 0, 0, 600000);
  const lgbEstimators = integerResource(input.resourceRequest?.lgb_estimators, sampleRows ? 400 : 1500, 80, 4000);
  const xgbEstimators = integerResource(input.resourceRequest?.xgb_estimators, sampleRows ? 400 : 1800, 80, 4000);
  const catIterations = integerResource(input.resourceRequest?.cat_iterations, sampleRows ? 400 : 2000, 80, 4000);
  const timeoutSeconds = integerResource(input.resourceRequest?.timeout_seconds ?? input.resourceRequest?.max_runtime_seconds, sampleRows ? 5400 : 43200, 600, 43200);
  const recoveryWaitSeconds = integerResource(input.resourceRequest?.recovery_wait_seconds, 1800, 0, 7200);
  const seeds = typeof input.resourceRequest?.seeds === "string" && input.resourceRequest.seeds.trim()
    ? input.resourceRequest.seeds.trim()
    : sampleRows ? "42" : "42";
  const xgbDevice = stringResource(input.resourceRequest?.xgb_device, "cpu", ["cpu", "cuda", "auto"]);
  const catTaskType = stringResource(input.resourceRequest?.cat_task_type, "CPU", ["CPU", "GPU", "auto"]);
  const gpuDeviceId = typeof input.resourceRequest?.gpu_device_id === "string" && input.resourceRequest.gpu_device_id.trim()
    ? input.resourceRequest.gpu_device_id.trim()
    : "auto";
  const normalizedResourceRequest = {
    ...(input.resourceRequest ?? {}),
    gpu: input.resourceRequest?.gpu ?? "available",
    task: "playground_series_s6e6",
    mode: "boosting_ensemble_lgb_xgb_cat",
    folds,
    seeds,
    sample_rows: sampleRows,
    xgb_device: xgbDevice,
    cat_task_type: catTaskType,
    gpu_device_id: gpuDeviceId,
    lgb_estimators: lgbEstimators,
    xgb_estimators: xgbEstimators,
    cat_iterations: catIterations,
    timeout_seconds: timeoutSeconds,
    recovery_wait_seconds: recoveryWaitSeconds
  };
  const localArtifactRoot = path.join(
    workspaceRoot,
    "workspace",
    "workstation_runs",
    input.taskId,
    input.runId ?? input.jobId,
    "hpc_gpu_training"
  );
  const script = path.join(workspaceRoot, "scripts", "run_hpc_kaggle_boosting_ensemble.py");
  const args = [
    script,
    "--host", config.host,
    "--port", config.port,
    "--user", config.username,
    "--password-env", "GPU_SSH_PASSWORD",
    "--remote-root", config.remoteWorkspace,
    "--local-artifact-dir", localArtifactRoot,
    "--remote-python", s6e6BoostingPythonPath(),
    "--timeout-seconds", String(timeoutSeconds),
    "--folds", String(folds),
    "--seeds", seeds,
    "--sample-rows", String(sampleRows),
    "--xgb-device", xgbDevice,
    "--cat-task-type", catTaskType,
    "--gpu-device-id", gpuDeviceId,
    "--lgb-estimators", String(lgbEstimators),
    "--xgb-estimators", String(xgbEstimators),
    "--cat-iterations", String(catIterations),
    "--recovery-wait-seconds", String(recoveryWaitSeconds)
  ];
  if (config.socksProxy.host) {
    args.push("--proxy-host", config.socksProxy.host, "--proxy-port", config.socksProxy.port);
  } else {
    args.push("--proxy-host", "");
  }

  const scriptArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.launcher.txt`, `${pythonCommand()} ${args.map((arg) => JSON.stringify(arg)).join(" ")}`);
  const preflightManifestPath = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}_manifest.json`, {
    schema: "academic_research_os.gpu_job_manifest.v1",
    job_id: input.jobId, task_id: input.taskId, workstation_run_id: input.runId,
    agent_id: input.agentId, gate_id: input.gateId, provider: "ssh_gateway",
    command_template: input.template,
    resource_request: normalizedResourceRequest,
    remote_workspace: config.remoteWorkspace,
    local_artifact_root: relativeFromRoot(localArtifactRoot),
    log_path: `workspace/workstation_runs/${input.taskId}/${input.runId ?? input.jobId}/hpc_gpu_training`,
    pullback_policy: "metrics_submission_report_probabilities_stdout_stderr",
    timeout_seconds: timeoutSeconds,
    cancel_record_path: `workspace/gpu/jobs/${input.jobId}_cancel.json`,
    status: "prepared",
    created_at: new Date().toISOString()
  });

  try {
    await updateGpuJobManifestStatus(preflightManifestPath, "running", {
      started_at: new Date().toISOString(),
      launcher_artifact: scriptArtifact
    });
    const { stdout, stderr } = await execFileAsync(pythonCommand(), args, {
      cwd: workspaceRoot, timeout: 1000 * (timeoutSeconds + 600), maxBuffer: 1024 * 1024 * 16,
      env: { ...process.env, GPU_SSH_PASSWORD: config.password }
    });
    const stdoutArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.stdout.log`, stdout);
    const stderrArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.stderr.log`, stderr);
    const localRootRelative = relativeFromRoot(localArtifactRoot);
    const manifestArtifact = `${localRootRelative}/manifest.json`;
    const metricsArtifact = `${localRootRelative}/metrics.json`;
    const submissionArtifact = `${localRootRelative}/submission.csv`;
    const probabilitiesArtifact = `${localRootRelative}/oof_and_test_probabilities.npz`;
    const reportArtifact = `${localRootRelative}/report.md`;
    const progressArtifact = `${localRootRelative}/progress.jsonl`;
    const pulledArtifacts = [stdoutArtifact, stderrArtifact];
    for (const candidate of [manifestArtifact, metricsArtifact, submissionArtifact, probabilitiesArtifact, reportArtifact, progressArtifact]) {
      if (await fileExists(candidate)) pulledArtifacts.push(candidate);
    }
    const artifacts = [
      await artifactDescriptor(scriptArtifact, {
        artifact_type: "gpu_job_launcher", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: "S6E6 boosting ensemble GPU job launched through workstation whitelist template.",
        gate_dependency: "hpc_execution_approval"
      }),
      await artifactDescriptor(preflightManifestPath, {
        artifact_type: "gpu_job_manifest", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: "GPU boosting ensemble job manifest includes task, run, agent, gate, resource and pullback.",
        gate_dependency: "hpc_execution_approval"
      }),
      await artifactDescriptor(stdoutArtifact, {
        artifact_type: "remote_stdout", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: "Boosting ensemble remote stdout was archived.", gate_dependency: "hpc_execution_approval"
      }),
      await artifactDescriptor(stderrArtifact, {
        artifact_type: "remote_stderr", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: "Boosting ensemble remote stderr was archived.", gate_dependency: "hpc_execution_approval"
      })
    ];
    for (const [artifactPath, artifactType, claim] of [
      [metricsArtifact, "boosting_ensemble_metrics", "Remote boosting ensemble metrics were pulled back."],
      [submissionArtifact, "boosting_ensemble_submission", "Remote boosting ensemble produced a Kaggle submission file."],
      [probabilitiesArtifact, "boosting_ensemble_probabilities", "Remote boosting ensemble OOF/test probabilities were pulled back for downstream blend gates."],
      [reportArtifact, "boosting_ensemble_hpc_report", "Remote boosting ensemble report was pulled back."],
      [progressArtifact, "boosting_ensemble_progress", "Remote boosting ensemble progress heartbeat was pulled back."]
    ] as const) {
      if (await fileExists(artifactPath)) {
        artifacts.push(await artifactDescriptor(artifactPath, {
          artifact_type: artifactType, created_by_agent: input.agentId, stage: "hpc_execution",
          claim_binding: claim, gate_dependency: "hpc_execution_approval"
        }));
      }
    }
    const artifactManifestPath = await writeArtifactManifestArtifact({
      taskId: input.taskId, runId: input.runId ?? input.jobId,
      relativePath: `workspace/gpu/jobs/${input.jobId}_artifact_manifest.json`,
      artifacts, source: "gpu_ssh_gateway_boosting_ensemble", extra: { job_id: input.jobId, pulled_artifacts: pulledArtifacts }
    });
    const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}.json`, {
      schema: "academic_research_os.gpu_job_result.v1",
      job_id: input.jobId, task_id: input.taskId, workstation_run_id: input.runId,
      agent_id: input.agentId, gate_id: input.gateId, template: input.template,
      provider: "ssh_gateway", status: "submitted", command_template: input.template,
      remote_workspace: config.remoteWorkspace, remote_python: s6e6BoostingPythonPath(), local_artifact_root: localRootRelative,
      host_key_policy: hostKeyPolicy(), proxy_policy: proxyPolicy(), auth_policy: authPolicy(),
      script_artifact: scriptArtifact, job_manifest_path: preflightManifestPath,
      stdout_artifact: stdoutArtifact, stderr_artifact: stderrArtifact,
      metrics_artifact: await fileExists(metricsArtifact) ? metricsArtifact : null,
      submission_artifact: await fileExists(submissionArtifact) ? submissionArtifact : null,
      probabilities_artifact: await fileExists(probabilitiesArtifact) ? probabilitiesArtifact : null,
      report_artifact: await fileExists(reportArtifact) ? reportArtifact : null,
      progress_artifact: await fileExists(progressArtifact) ? progressArtifact : null,
      artifact_manifest_path: artifactManifestPath, pulled_artifacts: pulledArtifacts,
      resource_request: normalizedResourceRequest,
      stdout, stderr, created_at: new Date().toISOString()
    });
    await updateGpuJobManifestStatus(preflightManifestPath, "submitted", {
      finished_at: new Date().toISOString(),
      result_artifact: artifact,
      pulled_artifacts: pulledArtifacts
    });
    await logAction({ action: "gpu_boosting_ensemble_job_submitted", taskId: input.taskId, runId: input.runId ?? undefined, message: "S6E6 boosting ensemble GPU training completed.", artifactPath: artifact, metadata: { job_id: input.jobId, template: input.template } });
    return {
      ok: true, configured: true, status: "submitted", provider: "ssh_gateway",
      host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, stdout, stderr,
      artifact_path: artifact, job_manifest_path: preflightManifestPath, stdout_artifact: stdoutArtifact,
      stderr_artifact: stderrArtifact,
      metrics_artifact: await fileExists(metricsArtifact) ? metricsArtifact : undefined,
      submission_artifact: await fileExists(submissionArtifact) ? submissionArtifact : undefined,
      probabilities_artifact: await fileExists(probabilitiesArtifact) ? probabilitiesArtifact : undefined,
      report_artifact: await fileExists(reportArtifact) ? reportArtifact : undefined,
      artifact_manifest_path: artifactManifestPath, pulled_artifacts: pulledArtifacts
    };
  } catch (error) {
    const execError = error as Error & { stdout?: string | Buffer; stderr?: string | Buffer; code?: unknown; signal?: unknown };
    const message = error instanceof Error ? error.message : "S6E6 boosting ensemble GPU job failed.";
    const stdoutText = typeof execError.stdout === "string" ? execError.stdout : execError.stdout ? execError.stdout.toString("utf-8") : "";
    const stderrText = typeof execError.stderr === "string" ? execError.stderr : execError.stderr ? execError.stderr.toString("utf-8") : "";
    const localRootRelative = relativeFromRoot(localArtifactRoot);
    const candidateFailureArtifacts = [
      `${localRootRelative}/remote_stdout.log`,
      `${localRootRelative}/remote_stderr.log`,
      `${localRootRelative}/manifest.json`,
      `${localRootRelative}/progress.jsonl`
    ];
    const pulledFailureArtifacts = [];
    for (const candidate of candidateFailureArtifacts) {
      if (await fileExists(candidate)) pulledFailureArtifacts.push(candidate);
    }
    const errorText = [
      message,
      execError.code !== undefined ? `exit_code=${String(execError.code)}` : "",
      execError.signal !== undefined ? `signal=${String(execError.signal)}` : "",
      stdoutText ? `\n--- child stdout ---\n${stdoutText.slice(-12000)}` : "",
      stderrText ? `\n--- child stderr ---\n${stderrText.slice(-12000)}` : "",
      pulledFailureArtifacts.length ? `\n--- local failure artifacts ---\n${pulledFailureArtifacts.join("\n")}` : ""
    ].filter(Boolean).join("\n");
    const errorArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.error.log`, errorText);
    const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}.json`, {
      schema: "academic_research_os.gpu_job_result.v1", job_id: input.jobId, task_id: input.taskId,
      workstation_run_id: input.runId, agent_id: input.agentId, gate_id: input.gateId,
      template: input.template, provider: "ssh_gateway", status: "failed",
      remote_workspace: config.remoteWorkspace, remote_python: s6e6BoostingPythonPath(), host_key_policy: hostKeyPolicy(),
      proxy_policy: proxyPolicy(), auth_policy: authPolicy(), script_artifact: scriptArtifact,
      job_manifest_path: preflightManifestPath, stderr_artifact: errorArtifact, error: message,
      child_exit_code: execError.code ?? null,
      child_signal: execError.signal ?? null,
      child_stdout_tail: stdoutText ? stdoutText.slice(-4000) : null,
      child_stderr_tail: stderrText ? stderrText.slice(-4000) : null,
      local_artifact_root: localRootRelative,
      failure_artifacts: pulledFailureArtifacts,
      created_at: new Date().toISOString()
    });
    await updateGpuJobManifestStatus(preflightManifestPath, "failed", {
      finished_at: new Date().toISOString(),
      result_artifact: artifact,
      error_artifact: errorArtifact
    });
    await logAction({ action: "gpu_boosting_ensemble_job_failed", taskId: input.taskId, runId: input.runId ?? undefined, message, artifactPath: artifact, metadata: { job_id: input.jobId, template: input.template } });
    return { ok: true, configured: true, status: "failed", provider: "ssh_gateway", host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, error: message, artifact_path: artifact, job_manifest_path: preflightManifestPath, stderr_artifact: errorArtifact, pulled_artifacts: [errorArtifact, ...pulledFailureArtifacts] };
  }
}

function integerResource(value: unknown, fallback: number, min: number, max: number) {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  if (!Number.isFinite(numeric)) return fallback;
  return Math.max(min, Math.min(max, Math.trunc(numeric)));
}

function stringResource<T extends string>(value: unknown, fallback: T, allowed: readonly T[]) {
  const text = typeof value === "string" ? value.trim() : "";
  return allowed.includes(text as T) ? (text as T) : fallback;
}

async function submitPlaygroundS6E6LgbmOptunaGpuJob(input: {
  jobId: string;
  taskId: string;
  runId: string | null;
  agentId: string;
  gateId: string | null;
  template: string;
  resourceRequest?: Record<string, unknown>;
}): Promise<GpuGatewayResult> {
  const config = gpuSshConfig();
  const localArtifactRoot = path.join(
    workspaceRoot,
    "workspace",
    "workstation_runs",
    input.taskId,
    input.runId ?? input.jobId,
    "hpc_lgbm_optuna"
  );
  const trials = integerResource(input.resourceRequest?.trials, 8, 1, 80);
  const folds = integerResource(input.resourceRequest?.folds, 5, 2, 5);
  const sampleRows = integerResource(input.resourceRequest?.sample_rows, 0, 0, 600000);
  const nEstimators = integerResource(input.resourceRequest?.n_estimators, sampleRows ? 320 : 2500, 80, 4000);
  const earlyStoppingRounds = integerResource(input.resourceRequest?.early_stopping_rounds, sampleRows ? 40 : 100, 10, 250);
  const timeoutSeconds = integerResource(input.resourceRequest?.timeout_seconds, sampleRows ? 3600 : 28800, 600, 43200);
  const nJobs = integerResource(input.resourceRequest?.n_jobs, -1, -1, 64);
  const seed = integerResource(input.resourceRequest?.seed, 260612, 1, 2147483647);
  const script = path.join(workspaceRoot, "scripts", "run_hpc_exp016_lgbm_optuna_search.py");
  const args = [
    script,
    "--host", config.host,
    "--port", config.port,
    "--user", config.username,
    "--password-env", "GPU_SSH_PASSWORD",
    "--remote-root", config.remoteWorkspace,
    "--python-executable", s6e6BoostingPythonPath(),
    "--timeout-seconds", String(timeoutSeconds),
    "--trials", String(trials),
    "--folds", String(folds),
    "--seed", String(seed),
    "--sample-rows", String(sampleRows),
    "--n-estimators", String(nEstimators),
    "--early-stopping-rounds", String(earlyStoppingRounds),
    "--n-jobs", String(nJobs)
  ];
  if (config.socksProxy.host) {
    args.push("--proxy-host", config.socksProxy.host, "--proxy-port", config.socksProxy.port);
  } else {
    args.push("--proxy-host", "");
  }

  const scriptArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.launcher.txt`, `${pythonCommand()} ${args.map((arg) => JSON.stringify(arg)).join(" ")}`);
  const preflightManifestPath = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}_manifest.json`, {
    schema: "academic_research_os.gpu_job_manifest.v1",
    job_id: input.jobId, task_id: input.taskId, workstation_run_id: input.runId,
    agent_id: input.agentId, gate_id: input.gateId, provider: "ssh_gateway",
    command_template: input.template,
    resource_request: {
      gpu: "any_available",
      task: "playground_series_s6e6",
      mode: sampleRows ? "lgbm_optuna_dryrun" : "lgbm_optuna_full_search",
      trials,
      folds,
      sample_rows: sampleRows,
      n_estimators: nEstimators,
      early_stopping_rounds: earlyStoppingRounds,
      allow_evidence_only: input.resourceRequest?.allow_evidence_only === true
    },
    remote_workspace: config.remoteWorkspace,
    local_artifact_root: relativeFromRoot(localArtifactRoot),
    log_path: `workspace/workstation_runs/${input.taskId}/${input.runId ?? input.jobId}/hpc_lgbm_optuna`,
    pullback_policy: "metrics_trials_probabilities_preview_report_stdout_stderr",
    timeout_seconds: timeoutSeconds,
    cancel_record_path: `workspace/gpu/jobs/${input.jobId}_cancel.json`,
    status: "prepared",
    official_submission_started: false,
    created_at: new Date().toISOString()
  });

  try {
    const { stdout, stderr } = await execFileAsync(pythonCommand(), args, {
      cwd: workspaceRoot, timeout: (timeoutSeconds + 300) * 1000, maxBuffer: 1024 * 1024 * 24,
      env: { ...process.env, GPU_SSH_PASSWORD: config.password }
    });
    const stdoutArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.stdout.log`, stdout);
    const stderrArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.stderr.log`, stderr);
    let localRootRelative = relativeFromRoot(localArtifactRoot);
    try {
      const parsed = parseJsonObjectFromOutput(stdout) as { local_artifact_dir?: string };
      if (parsed.local_artifact_dir) localRootRelative = parsed.local_artifact_dir;
    } catch {
      // The launcher still writes stdout/stderr artifacts; keep the prepared path if JSON parsing fails.
    }
    const manifestArtifact = `${localRootRelative}/manifest.json`;
    const metricsArtifact = `${localRootRelative}/metrics.json`;
    const trialsArtifact = `${localRootRelative}/trial_results_compact.csv`;
    const probabilitiesArtifact = `${localRootRelative}/oof_and_test_probabilities.npz`;
    const previewArtifact = `${localRootRelative}/submission_preview_not_submitted.csv`;
    const reportArtifact = `${localRootRelative}/report.md`;
    const pulledArtifacts = [stdoutArtifact, stderrArtifact];
    for (const candidate of [manifestArtifact, metricsArtifact, trialsArtifact, probabilitiesArtifact, previewArtifact, reportArtifact]) {
      if (await fileExists(candidate)) pulledArtifacts.push(candidate);
    }
    const artifacts = [
      await artifactDescriptor(scriptArtifact, {
        artifact_type: "gpu_job_launcher", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: "EXP018 LightGBM Optuna challenger launched through workstation whitelist template.",
        gate_dependency: "hpc_execution_approval"
      }),
      await artifactDescriptor(preflightManifestPath, {
        artifact_type: "gpu_job_manifest", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: "Optuna GPU job manifest records bounded trials, folds, sample policy and pullback contract.",
        gate_dependency: "hpc_execution_approval"
      }),
      await artifactDescriptor(stdoutArtifact, {
        artifact_type: "remote_stdout", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: "Optuna remote stdout was archived.", gate_dependency: "hpc_execution_approval"
      }),
      await artifactDescriptor(stderrArtifact, {
        artifact_type: "remote_stderr", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: "Optuna remote stderr was archived.", gate_dependency: "hpc_execution_approval"
      })
    ];
    for (const [artifactPath, artifactType, claim] of [
      [metricsArtifact, "lgbm_optuna_metrics", "Remote Optuna metrics were pulled back."],
      [trialsArtifact, "lgbm_optuna_trials", "Remote Optuna trial table was pulled back."],
      [probabilitiesArtifact, "lgbm_optuna_probabilities", "Remote Optuna OOF/test probability artifact was pulled back."],
      [previewArtifact, "lgbm_optuna_submission_preview", "Remote Optuna prediction preview was pulled back but is not an official submission candidate."],
      [reportArtifact, "lgbm_optuna_hpc_report", "Remote Optuna report was pulled back."]
    ] as const) {
      if (await fileExists(artifactPath)) {
        artifacts.push(await artifactDescriptor(artifactPath, {
          artifact_type: artifactType, created_by_agent: input.agentId, stage: "hpc_execution",
          claim_binding: claim, gate_dependency: "hpc_execution_approval"
        }));
      }
    }
    const artifactManifestPath = await writeArtifactManifestArtifact({
      taskId: input.taskId, runId: input.runId ?? input.jobId,
      relativePath: `workspace/gpu/jobs/${input.jobId}_artifact_manifest.json`,
      artifacts, source: "gpu_ssh_gateway_lgbm_optuna", extra: { job_id: input.jobId, pulled_artifacts: pulledArtifacts }
    });
    const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}.json`, {
      schema: "academic_research_os.gpu_job_result.v1",
      job_id: input.jobId, task_id: input.taskId, workstation_run_id: input.runId,
      agent_id: input.agentId, gate_id: input.gateId, template: input.template,
      provider: "ssh_gateway", status: "submitted", command_template: input.template,
      experiment_id: "EXP018", runner: "hpc_lgbm_optuna_search",
      remote_workspace: config.remoteWorkspace, remote_python: s6e6BoostingPythonPath(), local_artifact_root: localRootRelative,
      host_key_policy: hostKeyPolicy(), proxy_policy: proxyPolicy(), auth_policy: authPolicy(),
      script_artifact: scriptArtifact, job_manifest_path: preflightManifestPath,
      stdout_artifact: stdoutArtifact, stderr_artifact: stderrArtifact,
      metrics_artifact: await fileExists(metricsArtifact) ? metricsArtifact : null,
      report_artifact: await fileExists(reportArtifact) ? reportArtifact : null,
      artifact_manifest_path: artifactManifestPath, pulled_artifacts: pulledArtifacts,
      official_submission_started: false,
      stdout, stderr, created_at: new Date().toISOString()
    });
    await logAction({ action: "gpu_lgbm_optuna_job_submitted", taskId: input.taskId, runId: input.runId ?? undefined, message: "S6E6 EXP018 LightGBM Optuna challenger completed through workstation GPU gateway.", artifactPath: artifact, metadata: { job_id: input.jobId, template: input.template, trials, folds, sample_rows: sampleRows } });
    return {
      ok: true, configured: true, status: "submitted", provider: "ssh_gateway",
      host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, stdout, stderr,
      artifact_path: artifact, job_manifest_path: preflightManifestPath, stdout_artifact: stdoutArtifact,
      stderr_artifact: stderrArtifact,
      metrics_artifact: await fileExists(metricsArtifact) ? metricsArtifact : undefined,
      report_artifact: await fileExists(reportArtifact) ? reportArtifact : undefined,
      artifact_manifest_path: artifactManifestPath, pulled_artifacts: pulledArtifacts
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "S6E6 EXP018 LightGBM Optuna GPU job failed.";
    const errorArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.error.log`, message);
    const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}.json`, {
      schema: "academic_research_os.gpu_job_result.v1", job_id: input.jobId, task_id: input.taskId,
      workstation_run_id: input.runId, agent_id: input.agentId, gate_id: input.gateId,
      template: input.template, provider: "ssh_gateway", status: "failed",
      experiment_id: "EXP018", runner: "hpc_lgbm_optuna_search",
      remote_workspace: config.remoteWorkspace, remote_python: s6e6BoostingPythonPath(),
      host_key_policy: hostKeyPolicy(), proxy_policy: proxyPolicy(), auth_policy: authPolicy(),
      script_artifact: scriptArtifact, job_manifest_path: preflightManifestPath,
      stderr_artifact: errorArtifact, error: message,
      official_submission_started: false,
      created_at: new Date().toISOString()
    });
    await logAction({ action: "gpu_lgbm_optuna_job_failed", taskId: input.taskId, runId: input.runId ?? undefined, message, artifactPath: artifact, metadata: { job_id: input.jobId, template: input.template } });
    return { ok: true, configured: true, status: "failed", provider: "ssh_gateway", host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, error: message, artifact_path: artifact, job_manifest_path: preflightManifestPath, stderr_artifact: errorArtifact, pulled_artifacts: [errorArtifact] };
  }
}

async function submitPlaygroundS6E6SingleModelGpuJob(input: {
  jobId: string;
  taskId: string;
  runId: string | null;
  agentId: string;
  gateId: string | null;
  template: string;
  resourceRequest?: Record<string, unknown>;
}): Promise<GpuGatewayResult> {
  const config = gpuSshConfig();
  const localArtifactRoot = path.join(
    workspaceRoot,
    "workspace",
    "workstation_runs",
    input.taskId,
    input.runId ?? input.jobId,
    "hpc_gpu_training"
  );
  const modelMap: Record<string, string> = {
    playground_s6e6_lightgbm: "lightgbm",
    playground_s6e6_xgboost: "xgboost",
    playground_s6e6_catboost: "catboost",
  };
  const modelName = modelMap[input.template] ?? "lightgbm";
  const folds = integerResource(input.resourceRequest?.folds, 5, 2, 5);
  const sampleRows = integerResource(input.resourceRequest?.sample_rows, 0, 0, 600000);
  const seed = integerResource(input.resourceRequest?.seed, 260612, 1, 2147483647);
  const timeoutSeconds = integerResource(input.resourceRequest?.timeout_seconds, sampleRows ? 3600 : 10800, 600, 43200);
  const seeds = typeof input.resourceRequest?.seeds === "string" && input.resourceRequest.seeds.trim()
    ? input.resourceRequest.seeds.trim()
    : sampleRows ? "42" : "42,3407,12345";
  const accelerator = typeof input.resourceRequest?.accelerator === "string" && ["auto", "cpu", "gpu"].includes(input.resourceRequest.accelerator)
    ? input.resourceRequest.accelerator
    : "auto";
  const classWeight = typeof input.resourceRequest?.class_weight === "string" && ["none", "half_balanced", "sqrt_balanced", "balanced", "strong_balanced"].includes(input.resourceRequest.class_weight)
    ? input.resourceRequest.class_weight
    : "none";
  const profile = typeof input.resourceRequest?.profile === "string" && ["default", "high_capacity", "conservative", "minority_recall"].includes(input.resourceRequest.profile)
    ? input.resourceRequest.profile
    : "default";
  const script = path.join(workspaceRoot, "scripts", "run_hpc_kaggle_single_model.py");
  const args = [
    script,
    "--model", modelName,
    "--host", config.host,
    "--port", config.port,
    "--user", config.username,
    "--password-env", "GPU_SSH_PASSWORD",
    "--remote-root", config.remoteWorkspace,
    "--python-executable", s6e6BoostingPythonPath(),
    "--local-artifact-dir", localArtifactRoot,
    "--timeout-seconds", String(timeoutSeconds),
    "--folds", String(folds),
    "--seeds", seeds,
    "--sample-rows", String(sampleRows),
    "--seed", String(seed),
    "--accelerator", accelerator,
    "--class-weight", classWeight,
    "--profile", profile
  ];
  if (config.socksProxy.host) {
    args.push("--proxy-host", config.socksProxy.host, "--proxy-port", config.socksProxy.port);
  } else {
    args.push("--proxy-host", "");
  }

  const scriptArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.launcher.txt`, `${pythonCommand()} ${args.map((arg) => JSON.stringify(arg)).join(" ")}`);
  const preflightManifestPath = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}_manifest.json`, {
    schema: "academic_research_os.gpu_job_manifest.v1",
    job_id: input.jobId, task_id: input.taskId, workstation_run_id: input.runId,
    agent_id: input.agentId, gate_id: input.gateId, provider: "ssh_gateway",
    command_template: input.template,
    resource_request: {
      gpu: "any_available",
      task: "playground_series_s6e6",
      mode: sampleRows ? "single_model_dryrun" : "single_model_full_training",
      model: modelName,
      accelerator,
      class_weight: classWeight,
      profile,
      folds,
      seeds,
      sample_rows: sampleRows,
      allow_evidence_only: input.resourceRequest?.allow_evidence_only === true
    },
    remote_workspace: config.remoteWorkspace,
    local_artifact_root: relativeFromRoot(localArtifactRoot),
    log_path: `workspace/workstation_runs/${input.taskId}/${input.runId ?? input.jobId}/hpc_gpu_training`,
    pullback_policy: "metrics_submission_probabilities_report_stdout_stderr",
    timeout_seconds: timeoutSeconds,
    cancel_record_path: `workspace/gpu/jobs/${input.jobId}_cancel.json`,
    status: "prepared",
    official_submission_started: false,
    created_at: new Date().toISOString()
  });

  try {
    const { stdout, stderr } = await execFileAsync(pythonCommand(), args, {
      cwd: workspaceRoot, timeout: (timeoutSeconds + 300) * 1000, maxBuffer: 1024 * 1024 * 24,
      env: { ...process.env, GPU_SSH_PASSWORD: config.password }
    });
    const stdoutArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.stdout.log`, stdout);
    const stderrArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.stderr.log`, stderr);
    const localRootRelative = relativeFromRoot(localArtifactRoot);
    const metricsArtifact = `${localRootRelative}/metrics.json`;
    const submissionArtifact = `${localRootRelative}/submission.csv`;
    const probabilitiesArtifact = `${localRootRelative}/oof_and_test_probabilities.npz`;
    const reportArtifact = `${localRootRelative}/report.md`;
    const pulledArtifacts = [stdoutArtifact, stderrArtifact];
    for (const candidate of [metricsArtifact, submissionArtifact, probabilitiesArtifact, reportArtifact]) {
      if (await fileExists(candidate)) pulledArtifacts.push(candidate);
    }
    const artifacts = [
      await artifactDescriptor(scriptArtifact, {
        artifact_type: "gpu_job_launcher", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: `S6E6 ${modelName} single-model GPU job launched through workstation whitelist template.`,
        gate_dependency: "hpc_execution_approval"
      }),
      await artifactDescriptor(preflightManifestPath, {
        artifact_type: "gpu_job_manifest", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: "GPU single-model job manifest includes task, run, agent, gate, resource and pullback.",
        gate_dependency: "hpc_execution_approval"
      }),
      await artifactDescriptor(stdoutArtifact, {
        artifact_type: "remote_stdout", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: "Single-model remote stdout was archived.", gate_dependency: "hpc_execution_approval"
      }),
      await artifactDescriptor(stderrArtifact, {
        artifact_type: "remote_stderr", created_by_agent: input.agentId, stage: "hpc_execution",
        claim_binding: "Single-model remote stderr was archived.", gate_dependency: "hpc_execution_approval"
      })
    ];
    for (const [artifactPath, artifactType, claim] of [
      [metricsArtifact, "single_model_metrics", "Remote single-model training metrics were pulled back."],
      [submissionArtifact, "single_model_submission", "Remote single-model training produced a Kaggle submission file."],
      [probabilitiesArtifact, "single_model_probabilities", "Remote single-model OOF/test probabilities were pulled back for later blend/selection agents."],
      [reportArtifact, "single_model_hpc_report", "Remote single-model training report was pulled back."]
    ] as const) {
      if (await fileExists(artifactPath)) {
        artifacts.push(await artifactDescriptor(artifactPath, {
          artifact_type: artifactType, created_by_agent: input.agentId, stage: "hpc_execution",
          claim_binding: claim, gate_dependency: "hpc_execution_approval"
        }));
      }
    }
    const artifactManifestPath = await writeArtifactManifestArtifact({
      taskId: input.taskId, runId: input.runId ?? input.jobId,
      relativePath: `workspace/gpu/jobs/${input.jobId}_artifact_manifest.json`,
      artifacts, source: "gpu_ssh_gateway_single_model", extra: { job_id: input.jobId, model: modelName, pulled_artifacts: pulledArtifacts }
    });
    const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}.json`, {
      schema: "academic_research_os.gpu_job_result.v1",
      job_id: input.jobId, task_id: input.taskId, workstation_run_id: input.runId,
      agent_id: input.agentId, gate_id: input.gateId, template: input.template,
      provider: "ssh_gateway", status: "submitted", command_template: input.template, model: modelName,
      remote_workspace: config.remoteWorkspace, local_artifact_root: localRootRelative,
      host_key_policy: hostKeyPolicy(), proxy_policy: proxyPolicy(), auth_policy: authPolicy(),
      script_artifact: scriptArtifact, job_manifest_path: preflightManifestPath,
      stdout_artifact: stdoutArtifact, stderr_artifact: stderrArtifact,
      metrics_artifact: await fileExists(metricsArtifact) ? metricsArtifact : null,
      submission_artifact: await fileExists(submissionArtifact) ? submissionArtifact : null,
      probabilities_artifact: await fileExists(probabilitiesArtifact) ? probabilitiesArtifact : null,
      report_artifact: await fileExists(reportArtifact) ? reportArtifact : null,
      artifact_manifest_path: artifactManifestPath, pulled_artifacts: pulledArtifacts,
      official_submission_started: false,
      stdout, stderr, created_at: new Date().toISOString()
    });
    await logAction({ action: "gpu_single_model_job_submitted", taskId: input.taskId, runId: input.runId ?? undefined, message: `S6E6 ${modelName} single-model GPU training completed.`, artifactPath: artifact, metadata: { job_id: input.jobId, template: input.template, model: modelName, accelerator, class_weight: classWeight, profile, folds, sample_rows: sampleRows } });
    return {
      ok: true, configured: true, status: "submitted", provider: "ssh_gateway",
      host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, stdout, stderr,
      artifact_path: artifact, job_manifest_path: preflightManifestPath, stdout_artifact: stdoutArtifact,
      stderr_artifact: stderrArtifact,
      metrics_artifact: await fileExists(metricsArtifact) ? metricsArtifact : undefined,
      submission_artifact: await fileExists(submissionArtifact) ? submissionArtifact : undefined,
      probabilities_artifact: await fileExists(probabilitiesArtifact) ? probabilitiesArtifact : undefined,
      report_artifact: await fileExists(reportArtifact) ? reportArtifact : undefined,
      artifact_manifest_path: artifactManifestPath, pulled_artifacts: pulledArtifacts
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : `S6E6 ${modelName} single-model GPU job failed.`;
    const errorArtifact = await writeTextArtifact(`workspace/gpu/jobs/${input.jobId}.error.log`, message);
    const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${input.jobId}.json`, {
      schema: "academic_research_os.gpu_job_result.v1", job_id: input.jobId, task_id: input.taskId,
      workstation_run_id: input.runId, agent_id: input.agentId, gate_id: input.gateId,
      template: input.template, provider: "ssh_gateway", status: "failed", model: modelName,
      remote_workspace: config.remoteWorkspace, host_key_policy: hostKeyPolicy(),
      proxy_policy: proxyPolicy(), auth_policy: authPolicy(), script_artifact: scriptArtifact,
      job_manifest_path: preflightManifestPath, stderr_artifact: errorArtifact, error: message,
      official_submission_started: false,
      created_at: new Date().toISOString()
    });
    await logAction({ action: "gpu_single_model_job_failed", taskId: input.taskId, runId: input.runId ?? undefined, message, artifactPath: artifact, metadata: { job_id: input.jobId, template: input.template, model: modelName } });
    return { ok: true, configured: true, status: "failed", provider: "ssh_gateway", host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, error: message, artifact_path: artifact, job_manifest_path: preflightManifestPath, stderr_artifact: errorArtifact, pulled_artifacts: [errorArtifact] };
  }
}

export async function submitGpuJob(input: {
  taskId?: string;
  template?: string;
  runId?: string;
  agentId?: string;
  gateId?: string;
  resourceRequest?: Record<string, unknown>;
}): Promise<GpuGatewayResult> {
  const taskId = normalizeTaskId(input.taskId ?? "house_prices");
  const templateCandidate = input.template ?? defaultTemplateForTask(taskId);
  const jobId = `gpu_${stamp()}_${Math.random().toString(36).slice(2, 8)}`;
  const runId = input.runId ?? null;
  const agentId = input.agentId ?? "HpcGpuExecutionAgent";
  const gateId = input.gateId ?? (runId ? `${runId}_hpc_execution_approval` : null);
  if (!isGpuTemplate(templateCandidate)) {
    const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${jobId}.json`, {
      job_id: jobId,
      task_id: taskId,
      requested_template: templateCandidate,
      provider: "ssh_gateway",
      status: "rejected",
      allowed_templates: Object.keys(allowedTemplates),
      created_at: new Date().toISOString()
    });
    await logAction({ action: "gpu_job_template_rejected", taskId, message: `GPU job template rejected: ${templateCandidate}.`, artifactPath: artifact, metadata: { job_id: jobId, requested_template: templateCandidate, allowed_templates: Object.keys(allowedTemplates) } });
    return { ok: true, configured: hasGpuSshConfig(), status: "rejected" as const, provider: "ssh_gateway" as const, allowed_templates: Object.keys(allowedTemplates), error: `Unsupported GPU template: ${templateCandidate}`, artifact_path: artifact };
  }
  const template = templateCandidate;
  const command = allowedTemplates[template];
  const config = gpuSshConfig();
  const isConnectionSmoke = template === "connection_smoke";
  const workspacePolicyBlock = await rejectWorkspacePolicyViolation("submit_gpu_job");
  if (workspacePolicyBlock) return workspacePolicyBlock;

  if (!isConnectionSmoke) {
    const missingContractFields = [
      input.runId ? null : "run_id",
      input.agentId ? null : "agent_id",
      input.gateId ? null : "gate_id",
      input.resourceRequest && typeof input.resourceRequest === "object" ? null : "resource_request"
    ].filter(Boolean) as string[];
    if (missingContractFields.length) {
      const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${jobId}.json`, {
        schema: "academic_research_os.gpu_job_result.v1",
        job_id: jobId,
        task_id: taskId,
        workstation_run_id: runId,
        agent_id: agentId,
        gate_id: gateId,
        template,
        provider: "ssh_gateway",
        status: "rejected",
        reason: "Non-smoke GPU jobs must be launched by an explicit workstation agent/run/gate contract.",
        missing_contract_fields: missingContractFields,
        required_fields: ["task_id", "run_id", "agent_id", "gate_id", "resource_request"],
        created_at: new Date().toISOString()
      });
      await logAction({
        action: "gpu_job_contract_rejected",
        taskId,
        runId: runId ?? undefined,
        message: "GPU job rejected because the workstation run/agent/gate/resource contract was incomplete.",
        artifactPath: artifact,
        metadata: { job_id: jobId, template, missing_contract_fields: missingContractFields }
      });
      return {
        ok: true,
        configured: hasGpuSshConfig(),
        status: "rejected" as const,
        provider: "ssh_gateway" as const,
        error: `Missing required GPU job fields: ${missingContractFields.join(", ")}`,
        artifact_path: artifact,
        allowed_templates: Object.keys(allowedTemplates),
        required_fields: ["task_id", "run_id", "agent_id", "gate_id", "resource_request"]
      };
    }
    if (taskId === "playground_series_s6e6" && template.startsWith("playground_s6e6_")) {
      const strategyGate = await evaluateStrategyExecutionGate({
        taskId,
        requestedTemplate: template,
        allowEvidenceOnly: input.resourceRequest?.allow_evidence_only === true
      });
      if (!strategyGate.gate.allowed_to_execute) {
        const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${jobId}.json`, {
          schema: "academic_research_os.gpu_job_result.v1",
          job_id: jobId,
          task_id: taskId,
          workstation_run_id: runId,
          agent_id: agentId,
          gate_id: gateId,
          template,
          provider: "ssh_gateway",
          status: "rejected",
          reason: "S6E6 GPU job rejected by strategy execution score gate.",
          strategy_execution_gate: strategyGate.gate,
          required_policy: "Use only score-improvement candidates for S6E6 unless allow_evidence_only=true is explicitly recorded in resource_request.",
          created_at: new Date().toISOString()
        });
        await logAction({
          action: "gpu_job_strategy_gate_rejected",
          taskId,
          runId: runId ?? undefined,
          message: `GPU job rejected by S6E6 strategy score gate: ${template}.`,
          artifactPath: artifact,
          metadata: {
            job_id: jobId,
            template,
            selected_template: strategyGate.gate.selected_template,
            blocked_reasons: strategyGate.gate.blocked_reasons
          }
        });
        return {
          ok: true,
          configured: hasGpuSshConfig(),
          status: "rejected" as const,
          provider: "ssh_gateway" as const,
          error: strategyGate.gate.blocked_reasons.join(" | ") || "S6E6 strategy execution gate blocked this GPU job.",
          artifact_path: artifact,
          allowed_templates: Object.keys(allowedTemplates)
        };
      }
    }
    const approvedGate = gateId
      ? await prisma.gate.findFirst({ where: { id: gateId, taskId, decision: "approved" } })
      : runId
        ? await prisma.gate.findFirst({ where: { taskId, runId, gateType: "hpc_execution_approval", decision: "approved" } })
        : null;
    if (!approvedGate) {
      const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${jobId}.json`, {
        job_id: jobId,
        task_id: taskId,
        workstation_run_id: runId,
        agent_id: agentId,
        template,
        provider: "ssh_gateway",
        status: "rejected",
        reason: "hpc_execution_approval gate is required before non-smoke GPU jobs.",
        required_gate_id: gateId ?? `${runId ?? "<run_id>"}_hpc_execution_approval`,
        created_at: new Date().toISOString()
      });
      await logAction({
        action: "gpu_job_gate_rejected",
        taskId,
        runId: runId ?? undefined,
        message: "GPU job rejected because hpc_execution_approval is not approved.",
        artifactPath: artifact,
        metadata: { job_id: jobId, template, required_gate_id: gateId }
      });
      return {
        ok: true,
        configured: hasGpuSshConfig(),
        status: "rejected" as const,
        provider: "ssh_gateway" as const,
        error: "hpc_execution_approval gate is required before non-smoke GPU jobs.",
        artifact_path: artifact,
        allowed_templates: Object.keys(allowedTemplates)
      };
    }
  }

  if (!hasGpuSshConfig()) {
    const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${jobId}.json`, {
      job_id: jobId,
      task_id: taskId,
      template,
      provider: "ssh_gateway",
      status: "not_configured",
      missing_env: missingEnv(),
      created_at: new Date().toISOString()
    });
    await logAction({ action: "gpu_job_not_configured", taskId, message: "GPU job was not submitted because SSH credentials are not configured.", artifactPath: artifact, metadata: { job_id: jobId, missing_env: missingEnv() } });
    return { ok: true, configured: false, status: "not_configured" as const, provider: "ssh_gateway" as const, missing_env: missingEnv(), artifact_path: artifact };
  }

  if (!isConnectionSmoke) {
    const connection = await testGpuConnection();
    if (connection.status !== "passed") {
      return writeFreshGpuSmokeBlocker({
        jobId,
        taskId,
        runId,
        agentId,
        gateId,
        template,
        connection
      });
    }
  }

  if (template === "playground_s6e6_pytorch_mlp") {
    return submitPlaygroundS6E6GpuJob({
      jobId,
      taskId,
      runId,
      agentId,
      gateId,
      resourceRequest: input.resourceRequest
    });
  }

  if (template === "playground_s6e6_ensemble") {
    return submitPlaygroundS6E6EnsembleGpuJob({
      jobId,
      taskId,
      runId,
      agentId,
      gateId,
      resourceRequest: input.resourceRequest
    });
  }

  if (template === "playground_s6e6_lightgbm" || template === "playground_s6e6_xgboost" || template === "playground_s6e6_catboost") {
    return submitPlaygroundS6E6SingleModelGpuJob({
      jobId,
      taskId,
      runId,
      agentId,
      gateId,
      template,
      resourceRequest: input.resourceRequest
    });
  }

  if (template === "playground_s6e6_boosting_ensemble") {
    return submitPlaygroundS6E6BoostingEnsembleGpuJob({
      jobId,
      taskId,
      runId,
      agentId,
      gateId,
      template,
      resourceRequest: input.resourceRequest
    });
  }

  if (template === "playground_s6e6_lgbm_optuna") {
    return submitPlaygroundS6E6LgbmOptunaGpuJob({
      jobId,
      taskId,
      runId,
      agentId,
      gateId,
      template,
      resourceRequest: input.resourceRequest
    });
  }

  const jobScript = `set -e\nmkdir -p workspace/gpu_jobs/${jobId}\n${command}\necho GPU_JOB_COMPLETED`;
  const scriptArtifact = await writeTextArtifact(`workspace/gpu/jobs/${jobId}.sh`, jobScript);
  const preflightManifestPath = await writeJsonArtifact(`workspace/gpu/jobs/${jobId}_manifest.json`, {
    schema: "academic_research_os.gpu_job_manifest.v1",
    job_id: jobId,
    task_id: taskId,
    workstation_run_id: runId,
    agent_id: agentId,
    gate_id: gateId,
    provider: "ssh_gateway",
    command_template: template,
    resource_request: input.resourceRequest ?? { mode: isConnectionSmoke ? "cuda_smoke" : "whitelist_job" },
    remote_workspace: config.remoteWorkspace,
    log_path: `workspace/gpu_jobs/${jobId}`,
    pullback_policy: "stdout_stderr_and_declared_artifacts",
    timeout_seconds: 1800,
    cancel_record_path: `workspace/gpu/jobs/${jobId}_cancel.json`,
    status: "prepared",
    created_at: new Date().toISOString()
  });
  try {
    const { stdout, stderr } = await runRemoteCommand(jobScript, 1000 * 60 * 30, 1024 * 1024 * 8);
    const stdoutArtifact = await writeTextArtifact(`workspace/gpu/jobs/${jobId}.stdout.log`, stdout);
    const stderrArtifact = await writeTextArtifact(`workspace/gpu/jobs/${jobId}.stderr.log`, stderr);
    const artifacts = [
      await artifactDescriptor(scriptArtifact, {
        artifact_type: "gpu_job_script",
        created_by_agent: agentId,
        stage: "hpc_execution",
        claim_binding: "GPU job was launched through the workstation whitelist template.",
        gate_dependency: isConnectionSmoke ? null : "hpc_execution_approval"
      }),
      await artifactDescriptor(preflightManifestPath, {
        artifact_type: "gpu_job_manifest",
        created_by_agent: agentId,
        stage: "hpc_execution",
        claim_binding: "GPU job has a bounded manifest with resource, log and cancellation policy.",
        gate_dependency: isConnectionSmoke ? null : "hpc_execution_approval"
      }),
      await artifactDescriptor(stdoutArtifact, {
        artifact_type: "remote_stdout",
        created_by_agent: agentId,
        stage: "hpc_execution",
        claim_binding: "Remote GPU stdout was archived as evidence.",
        gate_dependency: isConnectionSmoke ? null : "hpc_execution_approval"
      }),
      await artifactDescriptor(stderrArtifact, {
        artifact_type: "remote_stderr",
        created_by_agent: agentId,
        stage: "hpc_execution",
        claim_binding: "Remote GPU stderr was archived as evidence.",
        gate_dependency: isConnectionSmoke ? null : "hpc_execution_approval"
      })
    ];
    const artifactManifestPath = await writeArtifactManifestArtifact({
      taskId,
      runId: runId ?? jobId,
      relativePath: `workspace/gpu/jobs/${jobId}_artifact_manifest.json`,
      artifacts,
      source: "gpu_ssh_gateway",
      extra: { job_id: jobId, pulled_artifacts: [stdoutArtifact, stderrArtifact] }
    });
    const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${jobId}.json`, {
      schema: "academic_research_os.gpu_job_result.v1",
      job_id: jobId,
      task_id: taskId,
      workstation_run_id: runId,
      agent_id: agentId,
      gate_id: gateId,
      template,
      provider: "ssh_gateway",
      status: "submitted",
      command_template: template,
      remote_workspace: config.remoteWorkspace,
      host_key_policy: hostKeyPolicy(),
      proxy_policy: proxyPolicy(),
      auth_policy: authPolicy(),
      script_artifact: scriptArtifact,
      job_manifest_path: preflightManifestPath,
      stdout_artifact: stdoutArtifact,
      stderr_artifact: stderrArtifact,
      artifact_manifest_path: artifactManifestPath,
      pulled_artifacts: [stdoutArtifact, stderrArtifact],
      stdout,
      stderr,
      created_at: new Date().toISOString()
    });
    await logAction({ action: "gpu_job_submitted", taskId, runId: runId ?? undefined, message: `GPU SSH job completed through whitelist template: ${template}.`, artifactPath: artifact, metadata: { job_id: jobId, template, script_artifact: scriptArtifact, job_manifest_path: preflightManifestPath, artifact_manifest_path: artifactManifestPath } });
    return { ok: true, configured: true, status: "submitted" as const, provider: "ssh_gateway" as const, host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, stdout, stderr, artifact_path: artifact, job_manifest_path: preflightManifestPath, stdout_artifact: stdoutArtifact, stderr_artifact: stderrArtifact, artifact_manifest_path: artifactManifestPath, pulled_artifacts: [stdoutArtifact, stderrArtifact] };
  } catch (error) {
    const message = error instanceof Error ? error.message : "GPU SSH job failed.";
    const errorArtifact = await writeTextArtifact(`workspace/gpu/jobs/${jobId}.error.log`, message);
    const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${jobId}.json`, {
      schema: "academic_research_os.gpu_job_result.v1",
      job_id: jobId,
      task_id: taskId,
      workstation_run_id: runId,
      agent_id: agentId,
      gate_id: gateId,
      template,
      provider: "ssh_gateway",
      status: "failed",
      remote_workspace: config.remoteWorkspace,
      host_key_policy: hostKeyPolicy(),
      proxy_policy: proxyPolicy(),
      auth_policy: authPolicy(),
      script_artifact: scriptArtifact,
      job_manifest_path: preflightManifestPath,
      stderr_artifact: errorArtifact,
      error: message,
      created_at: new Date().toISOString()
    });
    await logAction({ action: "gpu_job_failed", taskId, runId: runId ?? undefined, message, artifactPath: artifact, metadata: { job_id: jobId, template, script_artifact: scriptArtifact, job_manifest_path: preflightManifestPath } });
    return { ok: true, configured: true, status: "failed" as const, provider: "ssh_gateway" as const, host: config.host, username: config.username, remote_workspace: config.remoteWorkspace, error: message, artifact_path: artifact, job_manifest_path: preflightManifestPath, stderr_artifact: errorArtifact, pulled_artifacts: [errorArtifact] };
  }
}

export async function readGpuJob(jobId: string) {
  const { readJsonFile, resolveWorkspacePath } = await import("@/lib/server/paths");
  return readJsonFile(resolveWorkspacePath(`workspace/gpu/jobs/${jobId}.json`));
}

export async function cancelGpuJob(jobId: string) {
  const artifact = await writeJsonArtifact(`workspace/gpu/jobs/${jobId}_cancel.json`, {
    job_id: jobId,
    provider: "ssh_gateway",
    status: "cancel_requested",
    note: "Remote process cancellation is only available for jobs launched with a persistent scheduler in the next phase.",
    created_at: new Date().toISOString()
  });
  await logAction({ action: "gpu_job_cancel_requested", message: `GPU job cancel request recorded: ${jobId}`, artifactPath: artifact, metadata: { job_id: jobId } });
  return { ok: true, configured: hasGpuSshConfig(), status: "cancel_requested" as const, provider: "ssh_gateway" as const, artifact_path: artifact };
}

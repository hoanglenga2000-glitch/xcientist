import fs from "node:fs";
import path from "node:path";
import { ProviderBoundaryError, resolveDeepSeekEndpoint } from "@/lib/security/provider-boundary";

export const supportedSecretEnvKeys = [
  "ANTHROPIC_API_KEY",
  "ANTHROPIC_API_KEY_FILE",
  "CLAUDE_API_KEY",
  "CLAUDE_API_KEY_FILE",
  "DEEPSEEK_API_KEY",
  "DEEPSEEK_API_KEY_FILE",
  "DEEPSEEK_ALLOWED_ORIGINS",
  "DEEPSEEK_BASE_URL",
  "DEEPSEEK_MODEL",
  "GPU_SSH_HOST",
  "GPU_SSH_HOST_FILE",
  "GPU_SSH_PORT",
  "GPU_SSH_USER",
  "GPU_SSH_USER_FILE",
  "GPU_SSH_PASSWORD",
  "GPU_SSH_PASSWORD_FILE",
  "GPU_SSH_KEY_PATH",
  "GPU_SSH_KEY_PATH_FILE",
  "GPU_SSH_KNOWN_HOSTS_PATH",
  "GPU_SSH_KNOWN_HOSTS_PATH_FILE",
  "GPU_SSH_SOCKS_HOST",
  "GPU_SSH_SOCKS_PORT",
  "GPU_SSH_SOCKS_USER",
  "GPU_SSH_SOCKS_USER_FILE",
  "GPU_SSH_SOCKS_PASSWORD",
  "GPU_SSH_SOCKS_PASSWORD_FILE",
  "GPU_REMOTE_WORKSPACE",
  "GPU_REMOTE_WORKSPACE_FILE",
  "WORKSTATION_SECRET_DIR"
] as const;

function readFileIfPresent(filePath?: string) {
  if (!filePath) return "";
  try {
    return fs.readFileSync(filePath, "utf-8").trim();
  } catch {
    return "";
  }
}

function secretDirFile(names: string[]) {
  const dir = process.env.WORKSTATION_SECRET_DIR;
  if (!dir) return "";
  for (const name of names) {
    const candidate = path.join(dir, name);
    if (fs.existsSync(candidate)) return candidate;
  }
  return "";
}

export function secretValue(key: string, aliases: string[] = []) {
  const keys = [key, ...aliases];
  for (const candidate of keys) {
    const direct = process.env[candidate];
    if (direct) return direct;
    const fileValue = readFileIfPresent(process.env[`${candidate}_FILE`]);
    if (fileValue) return fileValue;
  }
  const dirFile = secretDirFile(keys);
  return readFileIfPresent(dirFile);
}

function secretPath(key: string, names: string[] = []) {
  return process.env[key] || process.env[`${key}_FILE`] || secretDirFile([key, ...names]);
}

export function hasClaudeApiKey() {
  return Boolean(claudeApiKeyValue());
}

export function claudeApiKeyValue() {
  return secretValue("ANTHROPIC_API_KEY", ["CLAUDE_API_KEY"]);
}

export function claudeApiKeyStatus() {
  return hasClaudeApiKey() ? "configured" : "not_configured";
}

export function deepSeekApiKeyValue() {
  return secretValue("DEEPSEEK_API_KEY", ["DEEPSEEK_KEY"]);
}

export function hasDeepSeekApiKey() {
  return Boolean(deepSeekApiKeyValue());
}

export function deepSeekConfig() {
  const rawBaseUrl = process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com";
  let endpoint: ReturnType<typeof resolveDeepSeekEndpoint> | null = null;
  let boundaryError: string | null = null;
  try {
    endpoint = resolveDeepSeekEndpoint(rawBaseUrl);
  } catch (error) {
    boundaryError = error instanceof ProviderBoundaryError ? error.code : "provider_url_invalid";
  }
  return {
    apiKey: deepSeekApiKeyValue(),
    baseUrl: endpoint?.baseUrl ?? "invalid",
    chatCompletionsUrl: endpoint?.chatCompletionsUrl ?? "",
    model: process.env.DEEPSEEK_MODEL || "deepseek-v4-flash",
    boundaryError
  };
}

export function deepSeekApiKeyStatus() {
  if (!hasDeepSeekApiKey()) return "not_configured";
  return deepSeekConfig().boundaryError ? "invalid_endpoint" : "configured";
}

export function gpuSshConfig() {
  return {
    host: secretValue("GPU_SSH_HOST"),
    port: secretValue("GPU_SSH_PORT") || "22",
    username: secretValue("GPU_SSH_USER"),
    password: secretValue("GPU_SSH_PASSWORD", ["HPC_SSH_PASSWORD"]),
    keyPath: secretPath("GPU_SSH_KEY_PATH", ["GPU_SSH_PRIVATE_KEY", "gpu_ssh_private_key", "id_rsa"]),
    knownHostsPath: secretPath("GPU_SSH_KNOWN_HOSTS_PATH", ["GPU_SSH_KNOWN_HOSTS", "known_hosts"]),
    remoteWorkspace: secretValue("GPU_REMOTE_WORKSPACE"),
    socksProxy: {
      host: secretValue("GPU_SSH_SOCKS_HOST", ["HPC_SOCKS_HOST"]),
      port: secretValue("GPU_SSH_SOCKS_PORT", ["HPC_SOCKS_PORT"]) || "1080",
      username: secretValue("GPU_SSH_SOCKS_USER", ["HPC_SOCKS_USER"]),
      password: secretValue("GPU_SSH_SOCKS_PASSWORD", ["HPC_SOCKS_PASSWORD"])
    }
  };
}

export function hasGpuSshConfig() {
  const config = gpuSshConfig();
  return Boolean(config.host && config.username && (config.keyPath || config.password) && config.remoteWorkspace);
}

export function gpuSshStatus() {
  return hasGpuSshConfig() ? "configured" : "not_configured";
}

export function redactedSecretStatus(value: string | undefined) {
  return value ? "configured" : "not_configured";
}

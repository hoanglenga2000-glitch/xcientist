import { isIP } from "node:net";

const DNS_LABEL = /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$/;
const SSH_USERNAME = /^[A-Za-z0-9._-]{1,128}$/;
const SHELL_PATH_METACHARACTERS = /[\0\r\n"%&|<>^!]/;

export function requireNetworkHost(value: string, label = "network host") {
  if (!value || value !== value.trim() || value.length > 253 || /\s/.test(value)) {
    throw new Error(`Invalid ${label}`);
  }
  if (isIP(value)) return value;
  const labels = value.split(".");
  if (!labels.length || labels.some((item) => !DNS_LABEL.test(item))) {
    throw new Error(`Invalid ${label}`);
  }
  return value;
}

export function requireTcpPort(value: string, label = "TCP port") {
  if (!/^\d{1,5}$/.test(value)) throw new Error(`Invalid ${label}`);
  const port = Number(value);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error(`Invalid ${label}`);
  return String(port);
}

export function requireSshUsername(value: string, label = "SSH username") {
  if (!SSH_USERNAME.test(value)) throw new Error(`Invalid ${label}`);
  return value;
}

export function requireOptionalProxyUsername(value: string) {
  if (!value) return "";
  return requireSshUsername(value, "proxy username");
}

export function requireShellSafePath(value: string, label = "command path") {
  if (!value || SHELL_PATH_METACHARACTERS.test(value)) throw new Error(`Invalid ${label}`);
  return value;
}

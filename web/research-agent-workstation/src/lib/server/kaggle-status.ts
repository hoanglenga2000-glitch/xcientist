export type KaggleAuthStatus = "not_configured" | "configured_unverified" | "authenticated";

export type KaggleAuthState = {
  status: KaggleAuthStatus;
  configured: boolean;
  authenticated: boolean;
  evidenceCredentialStatus: string;
};

const DEFAULT_SMOKE_MAX_AGE_MS = 15 * 60 * 1000;

function evidenceIsFresh(report: Record<string, unknown> | null, nowMs: number, maxAgeMs: number) {
  if (typeof report?.generated_at !== "string") return false;
  const generatedMs = Date.parse(report.generated_at);
  if (!Number.isFinite(generatedMs)) return false;
  const ageMs = nowMs - generatedMs;
  return ageMs >= -2 * 60 * 1000 && ageMs <= maxAgeMs;
}

export function deriveKaggleAuthState(
  report: Record<string, unknown> | null,
  envConfigured: boolean,
  nowMs: number = Date.now(),
  maxAgeMs: number = DEFAULT_SMOKE_MAX_AGE_MS,
): KaggleAuthState {
  const evidenceCredentialStatus = typeof report?.credential_status === "string"
    ? report.credential_status
    : "not_configured";
  const authenticated = Boolean(
    envConfigured
    && evidenceIsFresh(report, nowMs, maxAgeMs)
    && report?.authenticated === true
    && report?.status === "passed"
    && evidenceCredentialStatus === "authenticated_real_api"
    && report?.verification_method === "dpapi_status_and_real_api_smoke"
    && report?.credential_installed === true,
  );
  const configured = Boolean(envConfigured);
  return {
    status: authenticated ? "authenticated" : configured ? "configured_unverified" : "not_configured",
    configured,
    authenticated,
    evidenceCredentialStatus,
  };
}

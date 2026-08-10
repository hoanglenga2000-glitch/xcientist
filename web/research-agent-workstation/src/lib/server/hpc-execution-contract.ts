export const MISSING_HPC_EXECUTION_CONTRACT = "Missing HPC execution contract";

export type HpcExecutionContract = {
  job_id: number;
  credential_profile: string;
  resource_profile: string;
  execution_backend: "hpc";
};

const STRICT_HPC_AMBIENT_OVERRIDE_KEYS = [
  "GPU_SSH_HOST", "GPU_SSH_HOST_FILE",
  "GPU_SSH_USER", "GPU_SSH_USER_FILE",
  "GPU_SSH_PASSWORD", "GPU_SSH_PASSWORD_FILE",
  "GPU_SSH_KEY_PATH", "GPU_SSH_KEY_PATH_FILE",
  "GPU_SSH_SOCKS_HOST", "GPU_SSH_SOCKS_HOST_FILE",
  "GPU_SSH_SOCKS_USER", "GPU_SSH_SOCKS_USER_FILE",
  "GPU_SSH_SOCKS_PASSWORD", "GPU_SSH_SOCKS_PASSWORD_FILE",
  "GPU_SSH_JUMP_HOST", "GPU_SSH_JUMP_HOST_FILE",
  "GPU_SSH_JUMP_USER", "GPU_SSH_JUMP_USER_FILE",
  "GPU_SSH_KNOWN_HOSTS_PATH", "GPU_SSH_KNOWN_HOSTS_PATH_FILE",
  "GPU_REMOTE_WORKSPACE", "GPU_REMOTE_WORKSPACE_FILE",
  "EVOMIND_HPC_EXPECTED_HOST_UUID", "EVOMIND_HPC_EXPECTED_HOST_UUID_FILE",
  "EVOMIND_HPC_EXPECTED_GPU_UUID", "EVOMIND_HPC_EXPECTED_GPU_UUID_FILE",
  "GPU_SSH_PORT", "GPU_SSH_SOCKS_PORT", "GPU_SSH_JUMP_PORT",
] as const;

const REQUIRED_FIELDS = [
  "job_id",
  "credential_profile",
  "resource_profile",
  "execution_backend",
] as const;

export class HpcExecutionContractError extends Error {
  readonly code: "missing_hpc_execution_contract" | "invalid_hpc_execution_contract";
  readonly missingFields: string[];
  readonly statusCode = 422;

  constructor(
    message: string,
    options: {
      code: "missing_hpc_execution_contract" | "invalid_hpc_execution_contract";
      missingFields?: string[];
    },
  ) {
    super(message);
    this.name = "HpcExecutionContractError";
    this.code = options.code;
    this.missingFields = options.missingFields ?? [];
  }
}

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

export function buildHpcSubprocessEnvironment(
  base: Record<string, string | undefined>,
  contract: HpcExecutionContract,
) {
  const environment = { ...base };
  for (const name of STRICT_HPC_AMBIENT_OVERRIDE_KEYS) delete environment[name];
  environment.EVOMIND_SIIM_HPC_JOB_ID = String(contract.job_id);
  environment.EVOMIND_HPC_CREDENTIAL_PROFILE = contract.credential_profile;
  environment.EVOMIND_HPC_RESOURCE_PROFILE = contract.resource_profile;
  environment.EVOMIND_EXECUTION_BACKEND = contract.execution_backend;
  return environment;
}

export function taskRequiresHpcExecutionContract(
  taskId: string,
  metadata: Record<string, unknown>,
) {
  const normalized = taskId.trim().toLowerCase().replaceAll("_", "-");
  const nested = record(metadata.hpc_execution_contract);
  const objective = typeof metadata.objective === "string"
    ? metadata.objective.normalize("NFKC").trim().toLowerCase()
    : "";
  const explicitlyLocalOnly = /(?:do\s+not|don't|without|no)\s+(?:use\s+)?hpc\b|(?:不使用|不要使用|禁用)\s*hpc/i.test(objective);
  const objectiveRequiresHpc = !explicitlyLocalOnly && (
    /\bhpc\b/i.test(objective)
    || /(?:remote|external)\s+gpu\b/i.test(objective)
    || /远程\s*(?:gpu|计算|训练)/i.test(objective)
    || /(?:不要|禁止|不得)\s*(?:使用)?\s*本地\s*gpu/i.test(objective)
  );
  return normalized === "siim-isic-melanoma-classification"
    || metadata.requires_hpc === true
    || metadata.execution_target === "hpc"
    || metadata.execution_backend === "hpc"
    || nested.execution_backend === "hpc"
    || objectiveRequiresHpc;
}

export function parseHpcExecutionContract(
  value: unknown,
  options: { required: boolean },
): HpcExecutionContract | null {
  const outer = record(value);
  const contract = Object.keys(record(outer.hpc_execution_contract)).length > 0
    ? record(outer.hpc_execution_contract)
    : outer;
  const hasAnyField = REQUIRED_FIELDS.some((field) => contract[field] !== undefined);
  if (!options.required && !hasAnyField) return null;

  const missingFields = REQUIRED_FIELDS.filter((field) => {
    const fieldValue = contract[field];
    return fieldValue === undefined || fieldValue === null || fieldValue === "";
  });
  if (missingFields.length > 0) {
    throw new HpcExecutionContractError(MISSING_HPC_EXECUTION_CONTRACT, {
      code: "missing_hpc_execution_contract",
      missingFields: [...missingFields],
    });
  }

  const jobId = Number(contract.job_id);
  const credentialProfile = typeof contract.credential_profile === "string"
    ? contract.credential_profile.trim()
    : "";
  const resourceProfile = typeof contract.resource_profile === "string"
    ? contract.resource_profile.trim()
    : "";
  const executionBackend = contract.execution_backend;
  if (
    !Number.isSafeInteger(jobId)
    || jobId <= 0
    || credentialProfile !== `job${jobId}`
    || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(resourceProfile)
    || executionBackend !== "hpc"
  ) {
    throw new HpcExecutionContractError("Invalid HPC execution contract", {
      code: "invalid_hpc_execution_contract",
    });
  }

  return {
    job_id: jobId,
    credential_profile: credentialProfile,
    resource_profile: resourceProfile,
    execution_backend: "hpc",
  };
}

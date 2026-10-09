import { cookies } from "next/headers";
import { SESSION_COOKIE, sessionPrincipal } from "@/lib/server/local-session";
import { requireActiveTenantBinding, tenantPrincipal } from "@/lib/server/tenant-byoa";

export type AssistantManagedHpcIdentity = {
  tenant_id: string;
  owner_principal_id: string;
  job_id: number;
  credential_profile: string;
  allocation_generation: number;
  profile_instance_id: string;
  allocation_binding_id: string;
};

export async function currentAssistantManagedHpcIdentity(): Promise<AssistantManagedHpcIdentity | null> {
  try {
    const principal = sessionPrincipal((await cookies()).get(SESSION_COOKIE)?.value);
    const identity = tenantPrincipal(principal);
    const binding = await requireActiveTenantBinding(identity);
    return {
      tenant_id: identity.tenantId,
      owner_principal_id: identity.username,
      job_id: binding.job_id,
      credential_profile: binding.credential_profile,
      allocation_generation: binding.allocation_generation,
      profile_instance_id: binding.profile_instance_id,
      allocation_binding_id: binding.allocation_binding_id,
    };
  } catch {
    return null;
  }
}

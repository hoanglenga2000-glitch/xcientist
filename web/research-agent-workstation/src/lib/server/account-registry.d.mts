export type ManagedAccount = {
  username: string; tenant_id: string; password_hash: string; role: 'user' | 'admin';
  enabled: 0 | 1; session_version: number;
};
export function accountRegistryEnabled(): boolean;
export function findAccount(username: string): ManagedAccount | null;

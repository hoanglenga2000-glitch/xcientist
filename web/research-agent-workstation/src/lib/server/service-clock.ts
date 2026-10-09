export type ClockAnchor={schema:string;service_instance_id:string;boot_id:string;windows_utc_epoch_ms:number;qpc_timestamp:number;qpc_frequency:number;created_at_utc:string;web_monotonic_origin:string;renewal?:ClockAnchor};
export function computeManagedUtcMs(anchor:ClockAnchor,instance:string,boot:string,hrtimeNs:bigint){
 if(anchor.schema!=="evomind.service_clock_anchor.v1"||anchor.service_instance_id!==instance||anchor.boot_id!==boot||anchor.web_monotonic_origin!=="system_qpc"||!Number.isSafeInteger(anchor.windows_utc_epoch_ms)||!Number.isSafeInteger(anchor.qpc_timestamp)||!Number.isSafeInteger(anchor.qpc_frequency)||anchor.qpc_frequency<=0||Date.parse(anchor.created_at_utc)!==anchor.windows_utc_epoch_ms)throw new Error("service_clock_anchor_invalid");
 const billion=BigInt(1_000_000_000),million=BigInt(1_000_000),anchorNs=BigInt(anchor.qpc_timestamp)*billion/BigInt(anchor.qpc_frequency),elapsedNs=hrtimeNs-anchorNs;
 if(elapsedNs<BigInt(0)||elapsedNs>BigInt(86_400_000_000_000))throw new Error("service_clock_anchor_invalid");
 return anchor.windows_utc_epoch_ms+Number(elapsedNs/million);
}
export type ManagedClockEnvironment={
 nodeEnv?:string;platform?:string;anchorPath?:string;instance?:string;boot?:string;qpc?:string;frequency?:string;epoch?:string
};
export async function managedUtcFromEnvironment(input:ManagedClockEnvironment,readAnchor:(path:string)=>Promise<ClockAnchor>,hrtimeNs:bigint,fallbackNow:()=>number){
 const present=[input.anchorPath,input.instance,input.boot];
 if(present.every((value)=>!value)){
  if(input.nodeEnv!=="production")return{utc_ms:fallbackNow(),source:"development_fallback" as const};
  throw new Error("service_clock_anchor_missing");
 }
 if(present.some((value)=>!value))throw new Error("service_clock_anchor_missing");
 const anchor=await readAnchor(input.anchorPath!);
 if(input.qpc!==String(anchor.qpc_timestamp)||input.frequency!==String(anchor.qpc_frequency)||input.epoch!==String(anchor.windows_utc_epoch_ms))throw new Error("service_clock_anchor_environment_mismatch");
 // The launch anchor stays pinned to the process environment. A privileged
 // watchdog may attach a fresh sample for that same service and boot, without
 // restarting active work or weakening the 24-hour freshness requirement.
 let current=anchor;
 if(anchor.renewal){
  const renewal=anchor.renewal;
  const originNs=BigInt(anchor.qpc_timestamp)*BigInt(1_000_000_000)/BigInt(anchor.qpc_frequency);
  computeManagedUtcMs(anchor,input.instance!,input.boot!,originNs);
  if(renewal.renewal||renewal.qpc_frequency!==anchor.qpc_frequency||!Number.isSafeInteger(renewal.qpc_timestamp)||renewal.qpc_timestamp<=anchor.qpc_timestamp)throw new Error("service_clock_renewal_invalid");
  const expectedEpoch=anchor.windows_utc_epoch_ms+(renewal.qpc_timestamp-anchor.qpc_timestamp)*1000/anchor.qpc_frequency;
  if(Math.abs(renewal.windows_utc_epoch_ms-expectedEpoch)>5000)throw new Error("service_clock_renewal_discontinuity");
  current=renewal;
 }
 return{utc_ms:computeManagedUtcMs(current,input.instance!,input.boot!,hrtimeNs),source:"managed_anchor" as const};
}

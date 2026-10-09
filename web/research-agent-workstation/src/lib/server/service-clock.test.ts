import assert from "node:assert/strict";import test from "node:test";
// @ts-expect-error Node --experimental-strip-types requires the explicit .ts extension.
import {computeManagedUtcMs,managedUtcFromEnvironment,type ClockAnchor} from "./service-clock.ts";
const base:ClockAnchor={schema:"evomind.service_clock_anchor.v1",service_instance_id:"service-a",boot_id:"boot-a",windows_utc_epoch_ms:1_700_000_000_000,qpc_timestamp:1_230_000_000,qpc_frequency:10_000_000,created_at_utc:"2023-11-14T22:13:20.000Z",web_monotonic_origin:"system_qpc"};
const anchorNs=BigInt(base.qpc_timestamp)*BigInt(1_000_000_000)/BigInt(base.qpc_frequency);
test("anchor epoch plus monotonic elapsed is stable across concurrent reads",()=>{const values=Array.from({length:32},()=>computeManagedUtcMs(base,"service-a","boot-a",anchorNs+BigInt(250_000_000)));assert.deepEqual(new Set(values),new Set([1_700_000_000_250]))});
test("service restart and system boot drift reject the prior anchor",()=>{assert.throws(()=>computeManagedUtcMs(base,"service-b","boot-a",anchorNs),/invalid/);assert.throws(()=>computeManagedUtcMs(base,"service-a","boot-b",anchorNs),/invalid/)});
test("missing malformed tampered and stale anchors fail closed",()=>{for(const patch of [{schema:"bad"},{windows_utc_epoch_ms:1},{qpc_frequency:0},{created_at_utc:"bad"},{web_monotonic_origin:"bad"}])assert.throws(()=>computeManagedUtcMs({...base,...patch} as ClockAnchor,"service-a","boot-a",anchorNs),/invalid/);assert.throws(()=>computeManagedUtcMs(base,"service-a","boot-a",anchorNs+BigInt(86_400_000_000_001)),/invalid/)});
const complete={nodeEnv:"production",platform:"linux",anchorPath:"anchor.json",instance:"service-a",boot:"boot-a",qpc:String(base.qpc_timestamp),frequency:String(base.qpc_frequency),epoch:String(base.windows_utc_epoch_ms)};
test("production uses the managed anchor even when platform is non-win32",async()=>{const result=await managedUtcFromEnvironment(complete,async()=>base,anchorNs+BigInt(250_000_000),()=>1);assert.deepEqual(result,{utc_ms:1_700_000_000_250,source:"managed_anchor"})});
test("production missing partial and tampered anchor inputs fail closed",async()=>{await assert.rejects(()=>managedUtcFromEnvironment({...complete,anchorPath:undefined},async()=>base,anchorNs,()=>1),/missing/);await assert.rejects(()=>managedUtcFromEnvironment({...complete,epoch:"1"},async()=>base,anchorNs,()=>1),/mismatch/);await assert.rejects(()=>managedUtcFromEnvironment({nodeEnv:"production",platform:"linux"},async()=>base,anchorNs,()=>1),/missing/)});
test("development and test fallback only when all three anchor identity values are absent",async()=>{for(const nodeEnv of ["development","test"]){assert.deepEqual(await managedUtcFromEnvironment({nodeEnv,platform:"linux"},async()=>base,anchorNs,()=>42),{utc_ms:42,source:"development_fallback"});await assert.rejects(()=>managedUtcFromEnvironment({nodeEnv,anchorPath:"anchor.json"},async()=>base,anchorNs,()=>42),/missing/)}});

function renewed(hours:number):ClockAnchor {
 const ms=hours*3600_000;
 return {...base,qpc_timestamp:base.qpc_timestamp+ms*base.qpc_frequency/1000,windows_utc_epoch_ms:base.windows_utc_epoch_ms+ms,created_at_utc:new Date(base.windows_utc_epoch_ms+ms).toISOString()};
}
test('a fresh privileged sample renews the same running instance after 24 hours',async()=>{
 const renewal=renewed(20);
 const now=anchorNs+BigInt(25*3600)*BigInt(1_000_000_000);
 const result=await managedUtcFromEnvironment(complete,async()=>({...base,renewal}),now,()=>{throw new Error('must not use wall-clock fallback')});
 assert.equal(result.utc_ms,base.windows_utc_epoch_ms+25*3600_000);
 assert.equal(result.source,'managed_anchor');
});
test('renewal preserves boot, environment, continuity and freshness boundaries',async()=>{
 const now=anchorNs+BigInt(25*3600)*BigInt(1_000_000_000);
 const valid=renewed(20);
 for(const renewal of [renewed(0),{...valid,boot_id:'other'},{...valid,service_instance_id:'other'},{...valid,qpc_frequency:1},{...valid,windows_utc_epoch_ms:valid.windows_utc_epoch_ms+60_000},{...valid,renewal:valid},renewed(30)]){
  await assert.rejects(managedUtcFromEnvironment(complete,async()=>({...base,renewal}),now,()=>42));
 }
 await assert.rejects(managedUtcFromEnvironment(complete,async()=>({...base,renewal:valid}),anchorNs+BigInt(45*3600)*BigInt(1_000_000_000),()=>42),/invalid/);
 await assert.rejects(managedUtcFromEnvironment({...complete,epoch:'1'},async()=>({...base,renewal:valid}),now,()=>42),/mismatch/);
});

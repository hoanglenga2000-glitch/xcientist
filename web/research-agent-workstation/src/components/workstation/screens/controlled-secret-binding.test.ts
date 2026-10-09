import assert from "node:assert/strict";
import test from "node:test";

const binding = await import(new URL("./controlled-secret-binding.ts", import.meta.url).href) as typeof import("./controlled-secret-binding");
const { isConfirmedControlledSecretBinding, isExactPendingWeatherAcceleration } = binding;

const fixedRun = {
  id: "run_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  status: "waiting_approval",
  approvals: [{
    status: "pending",
    tool_name: "competition_data_accelerate",
    normalized_arguments: { competition: "weather4cast", timeout_seconds: 1800 },
  }],
  controlled_secret_availability: { weather4cast_sftp: 1 },
};

test("one exact Weather acceleration approval targets the existing fixed Run", () => {
  assert.equal(isExactPendingWeatherAcceleration(fixedRun), true);
  assert.equal(
    isConfirmedControlledSecretBinding(fixedRun, fixedRun.id, "weather4cast_sftp", 1),
    true,
  );
});

test("controlled secret binding confirmation fails closed on Run approval or count drift", () => {
  assert.equal(isConfirmedControlledSecretBinding(fixedRun, "run_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "weather4cast_sftp", 1), false);
  assert.equal(isExactPendingWeatherAcceleration({ ...fixedRun, status: "cancelled" }), false);
  assert.equal(isExactPendingWeatherAcceleration({ ...fixedRun, approvals: [] }), false);
  assert.equal(isExactPendingWeatherAcceleration({ ...fixedRun, approvals: [{ ...fixedRun.approvals[0], status: "approved" }] }), false);
  assert.equal(isExactPendingWeatherAcceleration({ ...fixedRun, approvals: [{ ...fixedRun.approvals[0], normalized_arguments: { competition: "ariel_2025", timeout_seconds: 1800 } }] }), false);
  assert.equal(isExactPendingWeatherAcceleration({ ...fixedRun, approvals: [{ ...fixedRun.approvals[0], normalized_arguments: { competition: "weather4cast", timeout_seconds: 1800, path: "forbidden" } }] }), false);
  assert.equal(isConfirmedControlledSecretBinding({ ...fixedRun, controlled_secret_availability: {} }, fixedRun.id, "weather4cast_sftp", 1), false);
  assert.equal(isConfirmedControlledSecretBinding({ ...fixedRun, controlled_secret_availability: { weather4cast_sftp: 2 } }, fixedRun.id, "weather4cast_sftp", 1), false);
});

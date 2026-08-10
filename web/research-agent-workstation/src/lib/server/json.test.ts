import assert from "node:assert/strict";
import test from "node:test";

// @ts-expect-error Node's strip-types test runner requires the explicit .ts suffix.
import { sanitizeWorkstationSummary } from "./json.ts";


test("summary projection removes nested infrastructure and credential identities", () => {
  const projected = sanitizeWorkstationSummary({
    workspace_root: "D:\\private\\EvoMind",
    prompt_tokens: 2048,
    nested: {
      ssh_host: "10.120.1.7",
      ssh_port: 10022,
      host_uuid: "host-private-id",
      gpu_uuid: "GPU-private-id",
      profile_name: "job-private-profile",
      credential_status: "configured_dpapi",
      access_token: "test-top-secret",
      artifact_path: "workspace/evomind_runs/demo/artifact_manifest.json",
    },
  }) as Record<string, unknown>;

  assert.equal(projected.workspace_root, undefined);
  assert.equal(projected.prompt_tokens, 2048);
  const nested = projected.nested as Record<string, unknown>;
  assert.deepEqual(nested, {
    artifact_path: "workspace/evomind_runs/demo/artifact_manifest.json",
  });
  const serialized = JSON.stringify(projected);
  assert.doesNotMatch(serialized, /10\.120|private-id|private-profile|top-secret|dpapi/i);
});


test("summary projection keeps relative references while redacting paths and inline secrets", () => {
  const projected = sanitizeWorkstationSummary({
    relative: "experiments/evolution/run-1/summary.json",
    workspace_relative: "workspace/tasks/task-1/code/model.py",
    windows_absolute: "D:\\Users\\Research User\\private\\model.bin",
    posix_absolute: "/home/research/private/model.bin",
    other_posix_absolute: "/usr/local/bin/private-tool",
    error: "ENOENT opening 'D:\\Users\\Research User\\private\\model.bin'",
    note: "Read C:\\Users\\research\\secret.txt then host=10.120.4.8:10022; token=abcdef",
    base_url: "http://127.0.0.1:65068/v1",
    paper_url: "https://arxiv.org/abs/2607.28568",
  }) as Record<string, unknown>;

  assert.equal(projected.relative, "experiments/evolution/run-1/summary.json");
  assert.equal(projected.workspace_relative, "workspace/tasks/task-1/code/model.py");
  assert.equal(projected.windows_absolute, "[local path redacted]");
  assert.equal(projected.posix_absolute, "[local path redacted]");
  assert.equal(projected.other_posix_absolute, "[local path redacted]");
  assert.equal(projected.error, "ENOENT opening [local path redacted]");
  assert.equal(projected.base_url, undefined);
  assert.equal(projected.paper_url, "https://arxiv.org/abs/2607.28568");
  const serialized = JSON.stringify(projected);
  assert.doesNotMatch(serialized, /Users|10\.120\.4\.8|abcdef/);
  assert.match(serialized, /local path redacted|infrastructure identity redacted/);
});

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

// @ts-expect-error Node's strip-types test runner requires the explicit .ts suffix.
import { CANONICAL_HASH_SCHEMA, canonicalJson, sha256Canonical, verifyExperienceBoard } from "./evolution-integrity.ts";

type SignedBoard = Record<string, unknown> & {
  cards: Array<Record<string, unknown>>;
  append_order: string[];
};

type CanonicalFixture = {
  canonical_hash_schema: string;
  cases: Array<{
    name: string;
    value: unknown;
    expected_sha256: string;
    expected_canonical?: string;
  }>;
  equivalence_groups: string[][];
};

const canonicalFixture = JSON.parse(readFileSync(fileURLToPath(new URL(
  "../../../../../tests/fixtures/canonical_json_f64_v1.json",
  import.meta.url,
)), "utf8")) as CanonicalFixture;

function experienceContent(nodeId = "EXP000", overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema: "evomind.experience_mcgs.v1",
    task_id: "fixture",
    node_id: nodeId,
    parent_ids: [],
    operator: "Draft",
    method_family: "linear",
    code_hash: "a".repeat(64),
    data_hashes: ["b".repeat(64)],
    prompt_hash: "c".repeat(64),
    execution_hash: "d".repeat(64),
    execution_id: nodeId,
    status: "success",
    public_validation_score: 0.5,
    quality: 0.5,
    progress: 0,
    novelty: 1,
    metric_direction: "maximize",
    error_signature: "",
    structural_fingerprint: ["family:linear", "operator:Draft"],
    cost: {
      prompt_tokens: 10, completion_tokens: 5, total_tokens: 15,
      wall_seconds: 1, gpu_seconds: 0, estimated_cost_usd: 0
    },
    provenance: { source: "verified_execution", changes_summary: "linear baseline" },
    summary: "",
    ...overrides
  };
}

function signedBoard(contents: Array<Record<string, unknown>>): SignedBoard {
  const cards = contents.map((content) => ({
    card_id: `exp_${sha256Canonical(content).slice(0, 24)}`,
    ...content
  }));
  let chain = "0".repeat(64);
  for (const card of cards) {
    const content: Record<string, unknown> = { ...card };
    delete content.card_id;
    chain = sha256Canonical({
      previous: chain,
      card_id: card.card_id,
      card_hash: sha256Canonical(content)
    });
  }
  const aggregate = { scope: "task_global", card_count: cards.length };
  const hashPayload = {
    schema: "evomind.experience_mcgs.v1",
    hash_canonicalization: CANONICAL_HASH_SCHEMA,
    task_id: "fixture",
    append_order: cards.map((card) => String(card.card_id)),
    append_chain_head: chain,
    cards,
    task_global_aggregation: aggregate
  };
  return {
    ...hashPayload,
    append_only: true,
    deduplication_key: "canonical_card_hash",
    board_hash: sha256Canonical(hashPayload)
  };
}

function validBoard(): SignedBoard {
  return signedBoard([experienceContent()]);
}

function validCrossoverBoard(): SignedBoard {
  return signedBoard([
    experienceContent("EXP000"),
    experienceContent("EXP001", {
      parent_ids: ["EXP000"], operator: "Improve", method_family: "tree"
    }),
    experienceContent("EXP002", {
      parent_ids: ["EXP000", "EXP001"], operator: "Crossover", method_family: "ensemble"
    })
  ]);
}

test("canonical JSON number contract matches the shared Python/TypeScript fixture", () => {
  assert.equal(canonicalFixture.canonical_hash_schema, CANONICAL_HASH_SCHEMA);
  const observed = new Map<string, string>();
  for (const fixture of canonicalFixture.cases) {
    const encoded = canonicalJson(fixture.value);
    observed.set(fixture.name, encoded);
    assert.equal(sha256Canonical(fixture.value), fixture.expected_sha256, fixture.name);
    if (fixture.expected_canonical !== undefined) {
      assert.equal(encoded, fixture.expected_canonical, fixture.name);
    }
  }
  for (const group of canonicalFixture.equivalence_groups) {
    assert.equal(new Set(group.map((name) => observed.get(name))).size, 1);
  }
});

test("canonical JSON contract rejects non-finite, unsafe, cyclic, and non-JSON values", () => {
  for (const value of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 2 ** 53]) {
    assert.throws(() => canonicalJson(value), TypeError);
  }
  assert.throws(() => canonicalJson(undefined), TypeError);
  assert.throws(() => canonicalJson(BigInt(1)), TypeError);
  assert.throws(() => canonicalJson(new Set([1, 2])), TypeError);
  const cyclic: unknown[] = [];
  cyclic.push(cyclic);
  assert.throws(() => canonicalJson(cyclic), TypeError);
});

test("Experience Board verifier accepts canonical append-only evidence", () => {
  const result = verifyExperienceBoard(validBoard());
  assert.equal(result.status, "verified");
  assert.equal(result.board_hash_valid, true);
  assert.equal(result.card_hashes_valid, true);
  assert.equal(result.append_chain_valid, true);
  assert.equal(result.public_only_valid, true);
});

test("Experience Board verifier fails closed on tampering and private-value leakage", () => {
  const tampered = structuredClone(validBoard());
  tampered.cards[0].summary = "private grader says score 0.987";
  const result = verifyExperienceBoard(tampered);
  assert.equal(result.status, "failed");
  assert.equal(result.public_only_valid, false);
  assert.equal(result.card_hashes_valid, false);

  const unknown = structuredClone(validBoard());
  unknown.cards[0].operator = "Unknown";
  assert.equal(verifyExperienceBoard(unknown).status, "failed");
});

test("Experience Board verifier accepts a valid two-family Crossover lineage", () => {
  const result = verifyExperienceBoard(validCrossoverBoard());
  assert.equal(result.status, "verified");
  assert.deepEqual(result.errors, []);
});

test("Experience Board verifier rejects unknown, future, self, and cyclic parents with valid hashes", () => {
  const fixtures = [
    {
      board: signedBoard([
        experienceContent("EXP001", { parent_ids: ["EXP404"], operator: "Improve" })
      ]),
      error: /unknown parent EXP404/
    },
    {
      board: signedBoard([
        experienceContent("EXP001", { parent_ids: ["EXP000"], operator: "Improve" }),
        experienceContent("EXP000")
      ]),
      error: /future parent EXP000/
    },
    {
      board: signedBoard([
        experienceContent("EXP000", { parent_ids: ["EXP000"], operator: "Improve" })
      ]),
      error: /self parent relation/
    },
    {
      board: signedBoard([
        experienceContent("EXP000", { parent_ids: ["EXP001"], operator: "Improve" }),
        experienceContent("EXP001", { parent_ids: ["EXP000"], operator: "Debug" })
      ]),
      error: /lineage cycle detected/
    }
  ];

  for (const fixture of fixtures) {
    const result = verifyExperienceBoard(fixture.board);
    assert.equal(result.status, "failed");
    assert.equal(result.board_hash_valid, true);
    assert.equal(result.card_hashes_valid, true);
    assert.equal(result.append_chain_valid, true);
    assert.match(result.errors.join("; "), fixture.error);
  }
});

test("Experience Board verifier enforces Crossover parent count, success, and distinct families", () => {
  const oneParent = signedBoard([
    experienceContent("EXP000"),
    experienceContent("EXP001", {
      parent_ids: ["EXP000"], operator: "Crossover", method_family: "ensemble"
    })
  ]);
  assert.match(
    verifyExperienceBoard(oneParent).errors.join("; "),
    /exactly two distinct parents/
  );

  const failedParent = validCrossoverBoard();
  const failedParentContents = failedParent.cards.map((card) => {
    const content = { ...card };
    delete content.card_id;
    if (content.node_id === "EXP001") content.status = "failed";
    return content;
  });
  const failedResult = verifyExperienceBoard(signedBoard(failedParentContents));
  assert.equal(failedResult.status, "failed");
  assert.match(failedResult.errors.join("; "), /parent EXP001 is not successful/);

  const sameFamily = signedBoard([
    experienceContent("EXP000", { method_family: "tree" }),
    experienceContent("EXP001", {
      parent_ids: ["EXP000"], operator: "Improve", method_family: "TREE"
    }),
    experienceContent("EXP002", {
      parent_ids: ["EXP000", "EXP001"], operator: "Crossover", method_family: "ensemble"
    })
  ]);
  const familyResult = verifyExperienceBoard(sameFamily);
  assert.equal(familyResult.status, "failed");
  assert.match(familyResult.errors.join("; "), /distinct non-empty method families/);
});

test("Experience Board verifier enforces non-Crossover parent arity", () => {
  const draftWithParent = signedBoard([
    experienceContent("EXP000"),
    experienceContent("EXP001", { parent_ids: ["EXP000"] })
  ]);
  assert.match(verifyExperienceBoard(draftWithParent).errors.join("; "), /Draft node EXP001 must not declare parents/);

  const improveWithoutParent = signedBoard([
    experienceContent("EXP000", { operator: "Improve" })
  ]);
  assert.match(verifyExperienceBoard(improveWithoutParent).errors.join("; "), /Improve node EXP000 must have exactly one parent/);

  const debugWithTwoParents = signedBoard([
    experienceContent("EXP000"),
    experienceContent("EXP001"),
    experienceContent("EXP002", { parent_ids: ["EXP000", "EXP001"], operator: "Debug" })
  ]);
  assert.match(verifyExperienceBoard(debugWithTwoParents).errors.join("; "), /Debug node EXP002 must have exactly one parent/);
});

test("Experience Board verifier returns a failed result when canonicalization throws", () => {
  const malformed = { ...validBoard(), task_id: BigInt(1) };
  const result = verifyExperienceBoard(malformed);
  assert.equal(result.status, "failed");
  assert.equal(result.board_hash_valid, false);
  assert.equal(result.card_hashes_valid, false);
  assert.equal(result.append_chain_valid, false);
  assert.equal(result.public_only_valid, false);
  assert.deepEqual(result.errors, ["Experience Board integrity verifier failed closed"]);
});

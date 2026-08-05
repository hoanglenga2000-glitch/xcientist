import { createHash } from "node:crypto";
import { Buffer } from "node:buffer";

export const EXPERIENCE_OPERATORS = ["Draft", "Improve", "Debug", "Crossover"] as const;
export const CANONICAL_HASH_SCHEMA = "evomind.canonical_json.f64.v1";
export type IntegrityOperator = typeof EXPERIENCE_OPERATORS[number];

export type ExperienceIntegrity = {
  status: "verified" | "failed" | "not_present";
  board_hash_valid: boolean | null;
  card_hashes_valid: boolean | null;
  append_chain_valid: boolean | null;
  public_only_valid: boolean | null;
  errors: string[];
};

const PRIVATE_EXPERIENCE_RE = /(?:private[\s_-]*(?:grader|score|feedback|labels?|metric|evaluation|result)|leaderboard|official[\s_-]*(?:rank|score|medal))/i;

function recordOf(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function stringValue(value: unknown, maxLength = 500): string {
  return typeof value === "string" ? value.slice(0, maxLength) : "";
}

function stringArray(value: unknown, limit = 32): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string").slice(0, limit)
    : [];
}

function lineageId(value: unknown): string {
  if (typeof value !== "string" || !value || value.length > 100 || value.trim() !== value) return "";
  return value;
}

export function canonicalJson(value: unknown): string {
  const scalarText = (item: string, location: string): string => {
    for (let index = 0; index < item.length; index += 1) {
      const codeUnit = item.charCodeAt(index);
      if (codeUnit >= 0xd800 && codeUnit <= 0xdbff) {
        const low = item.charCodeAt(index + 1);
        if (!(low >= 0xdc00 && low <= 0xdfff)) {
          throw new TypeError(`Lone high surrogate is forbidden at ${location}.`);
        }
        index += 1;
      } else if (codeUnit >= 0xdc00 && codeUnit <= 0xdfff) {
        throw new TypeError(`Lone low surrogate is forbidden at ${location}.`);
      }
    }
    return item;
  };

  const numberIdentity = (item: number, location: string): string => {
    if (!Number.isFinite(item)) throw new TypeError(`Non-finite JSON number at ${location}.`);
    if (Number.isInteger(item) && !Number.isSafeInteger(item)) {
      throw new TypeError(`Unsafe integer-valued JSON number at ${location}.`);
    }
    const normalized = Object.is(item, -0) ? 0 : item;
    const bytes = Buffer.allocUnsafe(8);
    bytes.writeDoubleBE(normalized, 0);
    return `f64:${bytes.toString("hex")}`;
  };

  const active = new WeakSet<object>();
  const normalize = (item: unknown, location: string): unknown => {
    if (item === null) return { $null: true };
    if (typeof item === "boolean") return { $boolean: item };
    if (typeof item === "string") return { $string: scalarText(item, location) };
    if (typeof item === "number") return { $number: numberIdentity(item, location) };
    if (Array.isArray(item)) {
      if (active.has(item)) throw new TypeError(`Cyclic JSON array at ${location}.`);
      const ownKeys = Reflect.ownKeys(item);
      if (ownKeys.some((key) => typeof key !== "string")) {
        throw new TypeError(`Symbol array key at ${location}.`);
      }
      const expectedKeys = new Set(["length", ...Array.from({ length: item.length }, (_, index) => String(index))]);
      if (ownKeys.length !== expectedKeys.size || ownKeys.some((key) => !expectedKeys.has(String(key)))) {
        throw new TypeError(`Sparse or extended JSON array at ${location}.`);
      }
      for (let index = 0; index < item.length; index += 1) {
        const descriptor = Object.getOwnPropertyDescriptor(item, String(index));
        if (!descriptor?.enumerable || !("value" in descriptor)) {
          throw new TypeError(`Non-data JSON array entry at ${location}[${index}].`);
        }
      }
      active.add(item);
      try {
        return { $array: item.map((child, index) => normalize(child, `${location}[${index}]`)) };
      } finally {
        active.delete(item);
      }
    }
    if (item && typeof item === "object") {
      const prototype = Object.getPrototypeOf(item);
      if (prototype !== Object.prototype && prototype !== null) {
        throw new TypeError(`Non-plain JSON object at ${location}.`);
      }
      if (active.has(item)) throw new TypeError(`Cyclic JSON object at ${location}.`);
      const ownKeys = Reflect.ownKeys(item);
      if (ownKeys.some((key) => typeof key !== "string")) {
        throw new TypeError(`Symbol object key at ${location}.`);
      }
      const entries = ownKeys.map((rawKey) => {
        const key = scalarText(String(rawKey), `${location}.<key>`);
        const descriptor = Object.getOwnPropertyDescriptor(item, String(rawKey));
        if (!descriptor?.enumerable || !("value" in descriptor)) {
          throw new TypeError(`Non-data JSON property at ${location}.${key}.`);
        }
        return [key, descriptor.value] as const;
      });
      entries.sort(([left], [right]) => Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8")));
      active.add(item);
      try {
        return {
          $object: entries.map(([key, child]) => [key, normalize(child, `${location}.${key}`)])
        };
      } finally {
        active.delete(item);
      }
    }
    throw new TypeError(`Unsupported canonical JSON value at ${location}.`);
  };

  return JSON.stringify({ $canonical: [CANONICAL_HASH_SCHEMA, normalize(value, "root")] });
}

export function sha256Canonical(value: unknown): string {
  return createHash("sha256").update(canonicalJson(value), "utf-8").digest("hex");
}

function findPrivateExperienceValue(value: unknown, location = "root"): string | null {
  if (typeof value === "string") return PRIVATE_EXPERIENCE_RE.test(value) ? location : null;
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      const found = findPrivateExperienceValue(value[index], `${location}[${index}]`);
      if (found) return found;
    }
    return null;
  }
  if (value && typeof value === "object") {
    for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
      if (PRIVATE_EXPERIENCE_RE.test(key)) return `${location}.${key}`;
      const found = findPrivateExperienceValue(child, `${location}.${key}`);
      if (found) return found;
    }
  }
  return null;
}

type LineageEntry = {
  index: number;
  nodeId: string;
  parentIds: string[];
  operator: IntegrityOperator | "";
  card: Record<string, unknown>;
};

function validateExperienceLineage(cards: Record<string, unknown>[], errors: string[]): void {
  const entries: LineageEntry[] = [];
  const nodeById = new Map<string, LineageEntry>();

  for (let index = 0; index < cards.length; index += 1) {
    const card = cards[index];
    const nodeId = lineageId(card.node_id);
    const rawParentIds = card.parent_ids;
    const parentIds = Array.isArray(rawParentIds) ? rawParentIds.map(lineageId) : [];
    const operator = EXPERIENCE_OPERATORS.includes(card.operator as IntegrityOperator)
      ? card.operator as IntegrityOperator
      : "";

    if (!nodeId) errors.push(`invalid node_id at cards[${index}]`);
    if (!Array.isArray(rawParentIds) || parentIds.some((parentId) => !parentId)) {
      errors.push(`invalid parent_ids at cards[${index}]`);
    }
    const validParentIds = parentIds.filter(Boolean);
    if (new Set(validParentIds).size !== validParentIds.length) {
      errors.push(`duplicate parent_id at cards[${index}]`);
    }

    const entry = { index, nodeId, parentIds: validParentIds, operator, card } satisfies LineageEntry;
    entries.push(entry);
    if (nodeId) {
      if (nodeById.has(nodeId)) errors.push(`duplicate node_id ${nodeId}`);
      else nodeById.set(nodeId, entry);
    }
  }

  for (const entry of entries) {
    const { index, nodeId, parentIds, operator } = entry;
    const nodeLabel = nodeId || `cards[${index}]`;
    if (operator === "Draft" && parentIds.length !== 0) {
      errors.push(`Draft node ${nodeLabel} must not declare parents`);
    } else if ((operator === "Improve" || operator === "Debug") && parentIds.length !== 1) {
      errors.push(`${operator} node ${nodeLabel} must have exactly one parent`);
    } else if (operator === "Crossover" && (parentIds.length !== 2 || new Set(parentIds).size !== 2)) {
      errors.push(`Crossover node ${nodeLabel} must have exactly two distinct parents`);
    }

    const resolvedParents: LineageEntry[] = [];
    for (const parentId of parentIds) {
      if (parentId === nodeId) {
        errors.push(`self parent relation at node ${nodeLabel}`);
        continue;
      }
      const parent = nodeById.get(parentId);
      if (!parent) {
        errors.push(`unknown parent ${parentId} for node ${nodeLabel}`);
        continue;
      }
      resolvedParents.push(parent);
      if (parent.index >= index) errors.push(`future parent ${parentId} for node ${nodeLabel}`);
    }

    if (operator === "Crossover" && resolvedParents.length === 2) {
      for (const parent of resolvedParents) {
        if (parent.card.status !== "success") {
          errors.push(`Crossover node ${nodeLabel} parent ${parent.nodeId} is not successful`);
        }
      }
      const families = resolvedParents.map((parent) => stringValue(parent.card.method_family, 120).trim().toLowerCase());
      if (families.some((family) => !family) || new Set(families).size !== 2) {
        errors.push(`Crossover node ${nodeLabel} parents must have distinct non-empty method families`);
      }
    }
  }

  const visitState = new Map<string, 0 | 1 | 2>();
  let cycleReported = false;
  const visit = (nodeId: string): void => {
    visitState.set(nodeId, 1);
    const entry = nodeById.get(nodeId);
    for (const parentId of entry?.parentIds ?? []) {
      if (!nodeById.has(parentId)) continue;
      const state = visitState.get(parentId) ?? 0;
      if (state === 1) {
        if (!cycleReported) errors.push(`lineage cycle detected at node ${parentId}`);
        cycleReported = true;
      } else if (state === 0) {
        visit(parentId);
      }
    }
    visitState.set(nodeId, 2);
  };
  for (const nodeId of nodeById.keys()) {
    if ((visitState.get(nodeId) ?? 0) === 0) visit(nodeId);
  }
}

function verifyPresentExperienceBoard(board: Record<string, unknown>): ExperienceIntegrity {
  const errors: string[] = [];
  const cards = Array.isArray(board.cards) ? board.cards.map(recordOf) : [];
  const appendOrder = stringArray(board.append_order, 2048);
  if (!Array.isArray(board.cards)) errors.push("cards is not an array");
  if (!Array.isArray(board.append_order)
    || board.append_order.some((item) => typeof item !== "string" || !item)) {
    errors.push("append_order is not a string array");
  }
  const publicOnlyLocation = findPrivateExperienceValue(board);
  const publicOnlyValid = publicOnlyLocation === null;
  if (!publicOnlyValid) errors.push(`private evaluation material at ${publicOnlyLocation}`);
  if (board.append_only !== true) errors.push("append_only flag is not true");
  if (board.deduplication_key !== "canonical_card_hash") errors.push("deduplication_key mismatch");
  if (board.hash_canonicalization !== CANONICAL_HASH_SCHEMA) errors.push("hash_canonicalization mismatch");
  if (appendOrder.length !== cards.length) errors.push("append_order/card count mismatch");
  if (new Set(appendOrder).size !== appendOrder.length) errors.push("duplicate append_order card id");

  validateExperienceLineage(cards, errors);

  let cardHashesValid = true;
  for (let index = 0; index < cards.length; index += 1) {
    const card = cards[index];
    const cardId = stringValue(card.card_id, 100);
    if (!EXPERIENCE_OPERATORS.includes(card.operator as IntegrityOperator)) {
      cardHashesValid = false;
      errors.push(`unknown operator at cards[${index}]`);
    }
    const content = { ...card };
    delete content.card_id;
    const expected = `exp_${sha256Canonical(content).slice(0, 24)}`;
    if (!cardId || cardId !== expected || appendOrder[index] !== cardId) {
      cardHashesValid = false;
      errors.push(`card content hash mismatch at cards[${index}]`);
    }
  }

  let chainHead = "0".repeat(64);
  for (const card of cards) {
    const cardId = stringValue(card.card_id, 100);
    const content = { ...card };
    delete content.card_id;
    chainHead = sha256Canonical({ previous: chainHead, card_id: cardId, card_hash: sha256Canonical(content) });
  }
  const appendChainValid = stringValue(board.append_chain_head, 64) === chainHead;
  if (!appendChainValid) errors.push("append chain head mismatch");
  const hashPayload = {
    schema: board.schema,
    hash_canonicalization: board.hash_canonicalization,
    task_id: board.task_id,
    append_order: board.append_order,
    append_chain_head: board.append_chain_head,
    cards: board.cards,
    task_global_aggregation: board.task_global_aggregation
  };
  const boardHashValid = stringValue(board.board_hash, 64) === sha256Canonical(hashPayload);
  if (!boardHashValid) errors.push("board hash mismatch");
  return {
    status: errors.length ? "failed" : "verified",
    board_hash_valid: boardHashValid,
    card_hashes_valid: cardHashesValid,
    append_chain_valid: appendChainValid,
    public_only_valid: publicOnlyValid,
    errors: errors.slice(0, 16)
  };
}

export function verifyExperienceBoard(board: Record<string, unknown> | null): ExperienceIntegrity {
  if (board === null) return {
    status: "not_present", board_hash_valid: null, card_hashes_valid: null,
    append_chain_valid: null, public_only_valid: null, errors: []
  };
  try {
    if (!board || typeof board !== "object" || Array.isArray(board)) throw new TypeError("invalid board");
    return verifyPresentExperienceBoard(board);
  } catch {
    return {
      status: "failed",
      board_hash_valid: false,
      card_hashes_valid: false,
      append_chain_valid: false,
      public_only_valid: false,
      errors: ["Experience Board integrity verifier failed closed"]
    };
  }
}

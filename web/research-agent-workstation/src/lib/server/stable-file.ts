import { createHash, randomUUID } from "node:crypto";
import { constants as fsConstants } from "node:fs";
import { lstat, mkdir, open, realpath, rename, rm } from "node:fs/promises";
import path from "node:path";

export type StableTextFile = {
  text: string;
  sha256: string;
  size: number;
  birthtime: Date;
  mtime: Date;
};

function insideRoot(candidate: string, root: string) {
  const relative = path.relative(root, candidate);
  return relative === "" || (!path.isAbsolute(relative) && relative !== ".." && !relative.startsWith(`..${path.sep}`));
}

function samePath(left: string, right: string) {
  const normalize = (value: string) => {
    const resolved = path.resolve(value);
    return process.platform === "win32" ? resolved.toLowerCase() : resolved;
  };
  return normalize(left) === normalize(right);
}

async function canonicalUnlinkedDirectory(directory: string, label: string) {
  const resolved = path.resolve(directory);
  const stat = await lstat(resolved);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new Error(`${label} must be a non-symlink directory`);
  }
  const canonical = await realpath(resolved);
  if (!samePath(canonical, resolved)) {
    throw new Error(`${label} must not traverse a junction or symlink`);
  }
  return canonical;
}

export async function ensurePrivateDirectory(directory: string, allowedRoot: string) {
  const root = await canonicalUnlinkedDirectory(allowedRoot, "Private directory allowed root");
  const target = path.resolve(directory);
  if (!insideRoot(target, root)) {
    throw new Error("Private directory escapes the allowed root");
  }
  const relative = path.relative(root, target);
  let current = root;
  for (const segment of relative.split(path.sep).filter(Boolean)) {
    current = path.join(current, segment);
    try {
      await mkdir(current, { mode: 0o700 });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    }
    const canonical = await canonicalUnlinkedDirectory(current, "Private directory component");
    if (!insideRoot(canonical, root)) {
      throw new Error("Private directory component escapes the allowed root");
    }
  }
  return target;
}

export async function writeAtomicPrivateTextFile(
  filePath: string,
  text: string,
  options: { allowedRoot: string; maxBytes: number }
) {
  const bytes = Buffer.from(text, "utf8");
  if (bytes.byteLength > options.maxBytes) {
    throw new Error("Private text file exceeds the allowed byte limit");
  }
  const absolutePath = path.resolve(filePath);
  const parent = await ensurePrivateDirectory(path.dirname(absolutePath), options.allowedRoot);
  const tempPath = path.join(parent, `.${path.basename(absolutePath)}.${randomUUID()}.tmp`);
  const noFollow = typeof fsConstants.O_NOFOLLOW === "number" ? fsConstants.O_NOFOLLOW : 0;
  const handle = await open(
    tempPath,
    fsConstants.O_WRONLY | fsConstants.O_CREAT | fsConstants.O_EXCL | noFollow,
    0o600
  );
  try {
    await handle.writeFile(bytes);
    await handle.sync();
  } finally {
    await handle.close();
  }
  try {
    await canonicalUnlinkedDirectory(parent, "Private text file parent");
    await rename(tempPath, absolutePath);
    return await readStableRegularTextFile(absolutePath, options);
  } catch (error) {
    await rm(tempPath, { force: true }).catch(() => undefined);
    throw error;
  }
}

function sameFileIdentity(
  left: { dev: number | bigint; ino: number | bigint },
  right: { dev: number | bigint; ino: number | bigint }
) {
  return String(left.dev) === String(right.dev) && String(left.ino) === String(right.ino);
}

export async function readStableRegularTextFile(
  filePath: string,
  options: { allowedRoot: string; maxBytes: number }
): Promise<StableTextFile> {
  if (!Number.isSafeInteger(options.maxBytes) || options.maxBytes <= 0) {
    throw new Error("Stable file byte limit must be a positive safe integer");
  }
  const absolutePath = path.resolve(filePath);
  const allowedRoot = await canonicalUnlinkedDirectory(options.allowedRoot, "Stable file allowed root");
  const canonicalParent = await canonicalUnlinkedDirectory(path.dirname(absolutePath), "Stable file parent");
  if (!insideRoot(canonicalParent, allowedRoot)) {
    throw new Error("Stable file parent escapes the allowed root");
  }

  const noFollow = typeof fsConstants.O_NOFOLLOW === "number" ? fsConstants.O_NOFOLLOW : 0;
  const handle = await open(absolutePath, fsConstants.O_RDONLY | noFollow);
  try {
    const opened = await handle.stat();
    const pathBefore = await lstat(absolutePath);
    if (pathBefore.isSymbolicLink() || !pathBefore.isFile() || !sameFileIdentity(pathBefore, opened)) {
      throw new Error("Stable file must be a non-symlink regular file with stable identity");
    }
    if (pathBefore.nlink !== 1 || opened.nlink !== 1) {
      throw new Error("Stable file must not be hard-linked");
    }
    if (pathBefore.size > options.maxBytes) {
      throw new Error("Stable file exceeds the allowed byte limit");
    }
    if (!opened.isFile() || !sameFileIdentity(pathBefore, opened)) {
      throw new Error("Stable file identity changed during open");
    }
    if (opened.size > options.maxBytes) {
      throw new Error("Stable file exceeds the allowed byte limit");
    }
    const chunks: Buffer[] = [];
    let total = 0;
    while (total <= options.maxBytes) {
      const chunk = Buffer.allocUnsafe(Math.min(64 * 1024, options.maxBytes + 1 - total));
      const { bytesRead } = await handle.read(chunk, 0, chunk.length, null);
      if (bytesRead === 0) break;
      chunks.push(chunk.subarray(0, bytesRead));
      total += bytesRead;
    }
    if (total > options.maxBytes) {
      throw new Error("Stable file exceeds the allowed byte limit");
    }
    const bytes = Buffer.concat(chunks, total);
    let text: string;
    try {
      text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    } catch {
      throw new Error("Stable text file is not valid UTF-8");
    }
    const afterRead = await handle.stat();
    const pathAfter = await lstat(absolutePath);
    const canonicalParentAfter = await canonicalUnlinkedDirectory(path.dirname(absolutePath), "Stable file parent");
    const canonicalPathAfter = await realpath(absolutePath);
    if (
      pathAfter.isSymbolicLink()
      || !pathAfter.isFile()
      || pathAfter.nlink !== 1
      || !sameFileIdentity(opened, afterRead)
      || !sameFileIdentity(opened, pathAfter)
      || opened.size !== afterRead.size
      || opened.mtimeMs !== afterRead.mtimeMs
      || opened.ctimeMs !== afterRead.ctimeMs
      || !samePath(canonicalParentAfter, canonicalParent)
      || !samePath(canonicalPathAfter, absolutePath)
    ) {
      throw new Error("Stable file changed while it was being read");
    }
    return {
      text,
      sha256: createHash("sha256").update(bytes).digest("hex"),
      size: afterRead.size,
      birthtime: afterRead.birthtime,
      mtime: afterRead.mtime
    };
  } finally {
    await handle.close();
  }
}

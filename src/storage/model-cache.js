import { MODEL_CONFIG, modelFileUrl } from "../config/model-config.js";

const MODEL_REPO_PATH = `/${MODEL_CONFIG.id}/resolve/`;

async function openModelCache() {
  if (!("caches" in globalThis)) return null;
  try {
    return await caches.open(MODEL_CONFIG.cacheName);
  } catch (error) {
    console.warn("모델 캐시를 열 수 없습니다.", error);
    return null;
  }
}
export async function getModelCacheStatus() {
  const cache = await openModelCache();
  if (!cache) {
    return { supported: false, vision: false, text: false };
  }
  const [vision, text] = await Promise.all([
    cache.match(modelFileUrl(MODEL_CONFIG.files.vision.path)),
    cache.match(modelFileUrl(MODEL_CONFIG.files.text.path)),
  ]);
  return {
    supported: true,
    vision: Boolean(vision),
    text: Boolean(text),
  };
}

export async function clearModelCache(kind = "all") {
  const cache = await openModelCache();
  if (!cache) return { supported: false, deleted: 0 };

  const requests = await cache.keys();
  let deleted = 0;
  for (const request of requests) {
    const url = new URL(request.url);
    if (!url.pathname.includes(MODEL_REPO_PATH)) continue;
    const filename = url.pathname.split("/").at(-1) ?? "";
    const isVision = filename.startsWith("vision_model") || filename === "preprocessor_config.json";
    const isText =
      filename.startsWith("text_model") ||
      filename.startsWith("tokenizer") ||
      filename === "vocab.json" ||
      filename === "merges.txt" ||
      filename === "tokenizer_config.json" ||
      filename === "special_tokens_map.json";
    const shouldDelete =
      kind === "all" ||
      (kind === "vision" && isVision) ||
      (kind === "text" && isText);
    if (shouldDelete && (await cache.delete(request))) deleted += 1;
  }
  return { supported: true, deleted };
}

export async function requestPersistentStorage() {
  if (!navigator.storage?.persist) {
    return { supported: false, persisted: false };
  }
  try {
    return {
      supported: true,
      persisted: await navigator.storage.persist(),
    };
  } catch (error) {
    console.warn("영구 저장소 요청에 실패했습니다.", error);
    return { supported: true, persisted: false };
  }
}

export async function getStorageSummary() {
  if (!navigator.storage?.estimate) return null;
  try {
    return await navigator.storage.estimate();
  } catch {
    return null;
  }
}

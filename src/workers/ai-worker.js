import {
  AutoProcessor,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  RawImage,
  env,
} from "@huggingface/transformers";
import { loadCategoryEmbeddings } from "../ai/category-embeddings.js";
import {
  averageEmbeddings,
  l2Normalize,
  passesIndependentThreshold,
  rankCandidates,
  scoreCandidatesIndependently,
} from "../ai/similarity.js";
import {
  KO_EN,
  SEARCH_NEGATIVE_PROMPTS,
} from "../config/categories.js";
import { MODEL_CONFIG } from "../config/model-config.js";

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;
env.remoteHost = MODEL_CONFIG.remoteHost;
env.remotePathTemplate = MODEL_CONFIG.remotePathTemplate;

let categoryEmbeddingsUrl = null;
let categoryDataPromise = null;
let visionModelPromise = null;
let textModelPromise = null;
let visionModel = null;
let textModel = null;
let processor = null;
let tokenizer = null;
let testFailureModel = null;
const imageEmbeddings = new Map();
const lastReportedPercent = { vision: -1, text: -1 };

function postEvent(event, payload = {}) {
  self.postMessage({ event, ...payload });
}

function serializeError(error) {
  return {
    name: error?.name ?? "Error",
    message: error?.message ?? String(error),
    stack: error?.stack ?? "",
  };
}

function progressCallback(kind) {
  return (progress) => {
    if (progress.status !== "progress" || !progress.file?.endsWith(".onnx")) return;
    const percent = Math.floor(progress.progress ?? 0);
    if (percent === lastReportedPercent[kind] && percent !== 100) return;
    lastReportedPercent[kind] = percent;
    postEvent("model-progress", {
      kind,
      file: progress.file,
      loaded: progress.loaded ?? 0,
      total: progress.total ?? MODEL_CONFIG.files[kind].bytes,
      progress: progress.progress ?? 0,
    });
  };
}

async function loadVisionModel() {
  if (visionModel) return true;
  if (visionModelPromise) return visionModelPromise;
  visionModelPromise = (async () => {
    if (testFailureModel === "vision") {
      testFailureModel = null;
      throw new Error("테스트용 이미지 모델 로딩 실패");
    }
    postEvent("model-state", { kind: "vision", status: "loading" });
    [processor, visionModel] = await Promise.all([
      AutoProcessor.from_pretrained(MODEL_CONFIG.id, {
        revision: MODEL_CONFIG.revision,
      }),
      CLIPVisionModelWithProjection.from_pretrained(MODEL_CONFIG.id, {
        revision: MODEL_CONFIG.revision,
        dtype: MODEL_CONFIG.dtype,
        device: MODEL_CONFIG.device,
        progress_callback: progressCallback("vision"),
      }),
    ]);
    postEvent("model-state", { kind: "vision", status: "ready" });
    return true;
  })().catch((error) => {
    visionModel = null;
    processor = null;
    visionModelPromise = null;
    postEvent("model-state", {
      kind: "vision",
      status: "error",
      error: serializeError(error),
    });
    throw error;
  });
  return visionModelPromise;
}

async function loadTextModel() {
  if (textModel && tokenizer) return true;
  if (textModelPromise) return textModelPromise;
  textModelPromise = (async () => {
    if (testFailureModel === "text") {
      testFailureModel = null;
      throw new Error("테스트용 텍스트 모델 로딩 실패");
    }
    postEvent("model-state", { kind: "text", status: "loading" });
    [tokenizer, textModel] = await Promise.all([
      AutoTokenizer.from_pretrained(MODEL_CONFIG.id, {
        revision: MODEL_CONFIG.revision,
      }),
      CLIPTextModelWithProjection.from_pretrained(MODEL_CONFIG.id, {
        revision: MODEL_CONFIG.revision,
        dtype: MODEL_CONFIG.dtype,
        device: MODEL_CONFIG.device,
        progress_callback: progressCallback("text"),
      }),
    ]);
    postEvent("model-state", { kind: "text", status: "ready" });
    return true;
  })().catch((error) => {
    textModel = null;
    tokenizer = null;
    textModelPromise = null;
    postEvent("model-state", {
      kind: "text",
      status: "error",
      error: serializeError(error),
    });
    throw error;
  });
  return textModelPromise;
}

async function ensureCategoryData() {
  if (!categoryEmbeddingsUrl) throw new Error("카테고리 임베딩 URL이 설정되지 않았습니다.");
  categoryDataPromise ??= loadCategoryEmbeddings(categoryEmbeddingsUrl);
  return categoryDataPromise;
}

async function analyzeFiles(files) {
  await loadVisionModel();
  const startedAt = performance.now();
  let completed = 0;
  for (const item of files) {
    if (imageEmbeddings.has(item.id)) {
      completed += 1;
      continue;
    }
    const image = await RawImage.read(item.file);
    const inputs = await processor(image);
    const { image_embeds: imageEmbeds } = await visionModel(inputs);
    imageEmbeddings.set(item.id, l2Normalize(imageEmbeds.data));
    imageEmbeds.dispose?.();
    for (const value of Object.values(inputs)) value?.dispose?.();
    completed += 1;

    const elapsed = performance.now() - startedAt;
    const averageMs = elapsed / completed;
    postEvent("analysis-progress", {
      completed,
      total: files.length,
      averageMs,
      etaMs: averageMs * (files.length - completed),
    });
  }
  return { analyzed: completed, stored: imageEmbeddings.size };
}

async function classifyDefault(ids, selectedCategoryIds, threshold) {
  const data = await ensureCategoryData();
  const selected = new Set(selectedCategoryIds);
  const candidates = data.categories
    .filter((category) => selected.has(category.id))
    .map((category) => ({
      id: category.id,
      label: category.labelKo,
      minimumScore: category.minimumScore ?? 0.5,
      embedding: category.embedding,
      negativeEmbedding: category.negativeEmbedding,
      promptEmbeddings: category.promptEmbeddings,
      negativePromptEmbeddings: category.negativePromptEmbeddings,
    }));

  const results = [];
  for (const id of ids) {
    const embedding = imageEmbeddings.get(id);
    if (!embedding) throw new Error(`${id} 이미지 임베딩이 없습니다.`);
    const ranked = scoreCandidatesIndependently(embedding, candidates);
    const matches = ranked.filter((candidate) =>
      passesIndependentThreshold(candidate, threshold),
    );
    results.push({
      id,
      matches: matches.map(({ id: categoryId, label, score, similarity, margin }) => ({
        id: categoryId,
        label,
        score,
        similarity,
        margin,
      })),
      top3: ranked.slice(0, 3).map(({ id: candidateId, label, score }) => ({
        id: candidateId,
        label,
        score,
      })),
    });
  }
  return results;
}

function translatedQuery(query) {
  return KO_EN[query] ?? query;
}

async function embedPromptGroups(groups) {
  await loadTextModel();
  const flatPrompts = groups.flatMap((group) => [
    ...group.prompts,
    ...group.negativePrompts,
  ]);
  const inputs = tokenizer(flatPrompts, { padding: true, truncation: true });
  const { text_embeds: textEmbeds } = await textModel(inputs);
  const allEmbeddings = [];
  for (let row = 0; row < textEmbeds.dims[0]; row += 1) {
    const start = row * MODEL_CONFIG.embeddingDimension;
    allEmbeddings.push(
      l2Normalize(textEmbeds.data.slice(start, start + MODEL_CONFIG.embeddingDimension)),
    );
  }
  textEmbeds.dispose?.();
  for (const value of Object.values(inputs)) value?.dispose?.();

  let offset = 0;
  return groups.map((group) => {
    const positiveEmbeddings = allEmbeddings.slice(offset, offset + group.prompts.length);
    offset += group.prompts.length;
    const negativeEmbeddings = allEmbeddings.slice(
      offset,
      offset + group.negativePrompts.length,
    );
    offset += group.negativePrompts.length;
    return {
      id: group.id,
      label: group.label,
      embedding: averageEmbeddings(positiveEmbeddings),
      negativeEmbedding: averageEmbeddings(negativeEmbeddings),
      promptEmbeddings: positiveEmbeddings,
      negativePromptEmbeddings: negativeEmbeddings,
    };
  });
}

async function searchImages(ids, queries, threshold) {
  const queryGroups = queries.map((query, index) => {
    const translated = translatedQuery(query);
    return {
      id: `query-${index}`,
      label: query,
      prompts: [
        `a photo of ${translated}`,
        `an image containing ${translated}`,
        `a close-up photo of ${translated}`,
      ],
      negativePrompts: [
        `a photo without ${translated}`,
        `an image unrelated to ${translated}`,
        ...SEARCH_NEGATIVE_PROMPTS,
      ],
    };
  });
  queryGroups.push({
    id: "other",
    label: "기타",
    prompts: SEARCH_NEGATIVE_PROMPTS,
    negativePrompts: SEARCH_NEGATIVE_PROMPTS,
  });
  const candidates = await embedPromptGroups(queryGroups);
  const results = [];
  for (const id of ids) {
    const embedding = imageEmbeddings.get(id);
    if (!embedding) throw new Error(`${id} 이미지 임베딩이 없습니다.`);
    const ranked = rankCandidates(embedding, candidates);
    const top = ranked[0];
    const matches =
      top.id !== "other" && top.score >= threshold
        ? [
            {
              id: top.id,
              label: top.label,
              score: top.score,
              similarity: top.similarity,
              margin: null,
            },
          ]
        : [];
    results.push({
      id,
      matches: matches.map(({ id: categoryId, label, score, similarity, margin }) => ({
        id: categoryId,
        label,
        score,
        similarity,
        margin,
      })),
      top3: ranked.slice(0, 3).map(({ id: candidateId, label, score }) => ({
        id: candidateId,
        label,
        score,
      })),
    });
  }
  return results;
}

async function unloadModels(kind = "all") {
  if ((kind === "all" || kind === "vision") && visionModel) {
    await visionModel.dispose();
    visionModel = null;
    processor = null;
    visionModelPromise = null;
    imageEmbeddings.clear();
  }
  if ((kind === "all" || kind === "text") && textModel) {
    await textModel.dispose();
    textModel = null;
    tokenizer = null;
    textModelPromise = null;
  }
  return true;
}

self.onmessage = async (event) => {
  const { requestId, type, payload = {} } = event.data;
  try {
    let result;
    switch (type) {
      case "init":
        categoryEmbeddingsUrl = payload.categoryEmbeddingsUrl;
        testFailureModel = payload.testFailureModel ?? null;
        result = { ready: true };
        break;
      case "load-model":
        result = payload.kind === "vision" ? await loadVisionModel() : await loadTextModel();
        break;
      case "analyze-files":
        result = await analyzeFiles(payload.files);
        break;
      case "classify-default":
        result = await classifyDefault(
          payload.ids,
          payload.selectedCategoryIds,
          payload.threshold,
        );
        break;
      case "search-images":
        result = await searchImages(payload.ids, payload.queries, payload.threshold);
        break;
      case "clear-embeddings":
        imageEmbeddings.clear();
        result = true;
        break;
      case "unload-models":
        result = await unloadModels(payload.kind);
        break;
      default:
        throw new Error(`알 수 없는 Worker 요청: ${type}`);
    }
    self.postMessage({ requestId, ok: true, result });
  } catch (error) {
    self.postMessage({
      requestId,
      ok: false,
      error: serializeError(error),
    });
  }
};

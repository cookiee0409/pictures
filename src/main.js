import "./styles.css";
import { DEFAULT_CATEGORIES } from "./config/categories.js";
import {
  CATEGORY_EMBEDDINGS_VERSION,
  MODEL_CONFIG,
  TRANSFORMERS_VERSION,
} from "./config/model-config.js";
import {
  clearModelCache,
  getModelCacheStatus,
  getStorageSummary,
  requestPersistentStorage,
} from "./storage/model-cache.js";
import { downloadZip } from "./utils/zip.js";

const $ = (id) => document.getElementById(id);
const visibleCategories = DEFAULT_CATEGORIES.filter((category) => !category.hidden);
const categoryById = new Map(DEFAULT_CATEGORIES.map((category) => [category.id, category]));

let files = [];
let mode = "default";
let busy = false;
let textConsentGranted = false;
let worker = null;
let workerRequestId = 0;
let pendingWorkerRequests = new Map();
const embeddedIds = new Set();
const modelLoadPromises = { vision: null, text: null };
const modelState = {
  vision: { status: "idle", progress: 0, loaded: 0, error: null, cached: false },
  text: { status: "idle", progress: 0, loaded: 0, error: null, cached: false },
};

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "알 수 없음";
  if (bytes < 1_000_000) return `${(bytes / 1_000).toFixed(0)}KB`;
  return `${(bytes / 1_000_000).toFixed(1)}MB`;
}

function formatDuration(milliseconds) {
  if (!Number.isFinite(milliseconds) || milliseconds <= 0) return "";
  const seconds = Math.max(1, Math.round(milliseconds / 1000));
  if (seconds < 60) return `약 ${seconds}초`;
  return `약 ${Math.ceil(seconds / 60)}분`;
}

function createWorker() {
  worker?.terminate();
  for (const { reject } of pendingWorkerRequests.values()) {
    reject(new Error("AI Worker가 다시 시작되었습니다."));
  }
  pendingWorkerRequests = new Map();
  worker = new Worker(new URL("./workers/ai-worker.js", import.meta.url), { type: "module" });
  worker.addEventListener("message", handleWorkerMessage);
  worker.addEventListener("error", handleWorkerError);
  const testFailureModel =
    ["localhost", "127.0.0.1"].includes(location.hostname)
      ? new URLSearchParams(location.search).get("testModelError")
      : null;
  return callWorker("init", {
    categoryEmbeddingsUrl:
      `${import.meta.env.BASE_URL}data/category-embeddings.json` +
      `?v=${encodeURIComponent(CATEGORY_EMBEDDINGS_VERSION)}`,
    testFailureModel,
  });
}

function callWorker(type, payload = {}) {
  const requestId = ++workerRequestId;
  return new Promise((resolve, reject) => {
    pendingWorkerRequests.set(requestId, { resolve, reject });
    worker.postMessage({ requestId, type, payload });
  });
}

function handleWorkerMessage(event) {
  const message = event.data;
  if (message.event) {
    handleWorkerEvent(message);
    return;
  }
  const pending = pendingWorkerRequests.get(message.requestId);
  if (!pending) return;
  pendingWorkerRequests.delete(message.requestId);
  if (message.ok) {
    pending.resolve(message.result);
  } else {
    const error = new Error(message.error?.message ?? "Worker 작업에 실패했습니다.");
    error.name = message.error?.name ?? "Error";
    pending.reject(error);
  }
}

function handleWorkerEvent(message) {
  if (message.event === "model-progress") {
    const state = modelState[message.kind];
    state.status = state.cached ? "loading" : "downloading";
    state.progress = message.progress;
    state.loaded = message.loaded;
    renderModelState(message.kind);
  }
  if (message.event === "model-state") {
    const state = modelState[message.kind];
    state.status = message.status;
    state.error = message.error?.message ?? null;
    if (message.status === "ready") {
      state.progress = 100;
      state.loaded = MODEL_CONFIG.files[message.kind].bytes;
      refreshCacheStatus();
    }
    renderModelState(message.kind);
    refreshControls();
  }
  if (message.event === "analysis-progress") {
    const eta = formatDuration(message.etaMs);
    $("runStatus").innerHTML =
      `<span class="spinner" aria-hidden="true"></span> 사진 분석 중 ${message.completed}/${message.total}` +
      (eta ? ` · 남은 시간 ${eta}` : "");
  }
}

function handleWorkerError(event) {
  console.error("AI Worker 오류", event.error ?? event.message);
  embeddedIds.clear();
  for (const kind of ["vision", "text"]) {
    if (modelState[kind].status === "ready") modelState[kind].status = "idle";
  }
  $("runStatus").textContent = "⚠️ AI 작업 공간에 오류가 생겼습니다. 아래 버튼으로 다시 시작하세요.";
  $("restartWorkerBtn").classList.remove("hidden");
  busy = false;
  refreshControls();
}

function renderCategories() {
  const root = $("categoryChips");
  root.innerHTML = "";
  for (const category of visibleCategories) {
    const label = document.createElement("label");
    label.className = "category-chip";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = category.id;
    input.checked = true;
    input.addEventListener("change", refreshControls);
    const text = document.createElement("span");
    text.textContent = category.labelKo;
    label.append(input, text);
    root.appendChild(label);
  }
}

function selectedCategoryIds() {
  return [...document.querySelectorAll("#categoryChips input:checked")].map(
    (input) => input.value,
  );
}

function renderModelState(kind) {
  const state = modelState[kind];
  const file = MODEL_CONFIG.files[kind];
  const isVision = kind === "vision";
  const title = isVision ? "이미지 모델" : "자유 검색 모델";
  const action = $(`${kind}Action`);
  const status = $(`${kind}Status`);
  const progress = $(`${kind}Progress`);
  const bar = $(`${kind}Bar`);

  bar.style.width = `${state.progress}%`;
  progress.textContent =
    state.status === "downloading"
      ? `${formatBytes(state.loaded)} / ${formatBytes(file.bytes)} · ${Math.round(state.progress)}%`
      : formatBytes(file.bytes);

  action.disabled = ["downloading", "loading", "ready"].includes(state.status);
  action.textContent = isVision ? "이미지 모델 준비" : "자유 검색 모델 설치";

  if (state.status === "ready") {
    status.textContent = "✅ 사용 준비 완료 · 이 브라우저 세션에서 로드됨";
    status.className = "model-status ready";
    action.textContent = "준비 완료";
  } else if (state.status === "downloading") {
    status.textContent = `⬇️ ${title} 다운로드 중 · 사진은 전송되지 않습니다.`;
    status.className = "model-status";
  } else if (state.status === "loading") {
    status.textContent = state.cached
      ? "⚙️ 이 기기의 캐시에서 모델을 여는 중…"
      : "⚙️ 다운로드한 모델을 초기화하는 중…";
    status.className = "model-status";
  } else if (state.status === "error") {
    status.textContent = `⚠️ ${state.error} · 다른 기능은 계속 사용할 수 있습니다.`;
    status.className = "model-status error";
    action.disabled = false;
    action.textContent = "다시 시도";
  } else if (state.cached) {
    status.textContent = "💾 이 기기에 저장됨 · 필요할 때 캐시에서 불러옵니다.";
    status.className = "model-status ready";
  } else {
    status.textContent = isVision
      ? "대기 중 · 사진을 선택해도 자동 다운로드하지 않습니다."
      : "설치되지 않음 · 자유 검색에 동의할 때만 다운로드합니다.";
    status.className = "model-status";
  }
}

async function refreshCacheStatus() {
  const status = await getModelCacheStatus();
  modelState.vision.cached = status.vision;
  modelState.text.cached = status.text;
  renderModelState("vision");
  renderModelState("text");

  const storage = await getStorageSummary();
  const storageText = status.supported ? "브라우저 모델 캐시 사용 가능" : "모델 캐시 API 미지원";
  $("storageStatus").textContent = storage
    ? `${storageText} · 현재 사이트 저장공간 ${formatBytes(storage.usage ?? 0)}`
    : storageText;
}

async function prepareModel(kind) {
  if (modelState[kind].status === "ready") return true;
  if (modelLoadPromises[kind]) return modelLoadPromises[kind];

  await requestPersistentStorage();
  modelState[kind].status = modelState[kind].cached ? "loading" : "downloading";
  modelState[kind].error = null;
  renderModelState(kind);
  modelLoadPromises[kind] = callWorker("load-model", { kind })
    .then(() => {
      modelState[kind].status = "ready";
      modelState[kind].progress = 100;
      modelState[kind].loaded = MODEL_CONFIG.files[kind].bytes;
      renderModelState(kind);
      return true;
    })
    .catch((error) => {
      modelState[kind].status = "error";
      modelState[kind].error = error.message;
      renderModelState(kind);
      throw error;
    })
    .finally(() => {
      modelLoadPromises[kind] = null;
      refreshControls();
    });
  return modelLoadPromises[kind];
}

function setMode(nextMode) {
  mode = nextMode;
  for (const button of document.querySelectorAll(".mode-button")) {
    button.classList.toggle("active", button.dataset.mode === mode);
    button.setAttribute("aria-pressed", button.dataset.mode === mode ? "true" : "false");
  }
  $("defaultModePanel").classList.toggle("hidden", mode !== "default");
  $("searchModePanel").classList.toggle("hidden", mode !== "search");
  updateSearchConsent();
  refreshControls();
}

function updateSearchConsent() {
  const needsConsent =
    mode === "search" &&
    modelState.text.status !== "ready" &&
    !modelState.text.cached &&
    !textConsentGranted;
  $("searchConsent").classList.toggle("hidden", !needsConsent);
  $("searchReadyNote").classList.toggle("hidden", needsConsent);
  if (!needsConsent) {
    $("searchReadyNote").textContent =
      modelState.text.status === "ready"
        ? "자유 검색 모델이 준비되었습니다."
        : modelState.text.cached
          ? "자유 검색 모델이 이 기기에 저장되어 있습니다. 검색 시 캐시에서 불러옵니다."
          : "설치 동의가 확인되었습니다. 검색 실행 시 텍스트 모델을 준비합니다.";
  }
}

function addFiles(fileList) {
  const existingKeys = new Set(files.map(({ file }) => `${file.name}:${file.size}:${file.lastModified}`));
  for (const file of fileList) {
    if (!file.type.startsWith("image/")) continue;
    const key = `${file.name}:${file.size}:${file.lastModified}`;
    if (existingKeys.has(key)) continue;
    existingKeys.add(key);
    files.push({
      id: crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`,
      file,
      url: URL.createObjectURL(file),
    });
  }
  renderSelectedFiles();
  $("results").innerHTML = "";
  refreshControls();
}

function renderSelectedFiles() {
  $("fileStatus").textContent = files.length
    ? `사진 ${files.length}장 준비됨 · 원본은 브라우저 메모리에만 유지됩니다.`
    : "";
  const root = $("selectedGrid");
  root.innerHTML = "";
  for (const item of files.slice(0, 12)) {
    const image = document.createElement("img");
    image.src = item.url;
    image.alt = item.file.name;
    image.loading = "lazy";
    root.appendChild(image);
  }
  if (files.length > 12) {
    const more = document.createElement("div");
    more.className = "more-files";
    more.textContent = `+${files.length - 12}`;
    root.appendChild(more);
  }
}

async function ensureImageEmbeddings() {
  const pendingFiles = files.filter((item) => !embeddedIds.has(item.id));
  if (!pendingFiles.length) return;
  await prepareModel("vision");
  await callWorker("analyze-files", {
    files: pendingFiles.map(({ id, file }) => ({ id, file })),
  });
  for (const item of pendingFiles) embeddedIds.add(item.id);
}

function parseQueries() {
  return $("searchInput")
    .value.split(",")
    .map((query) => query.trim())
    .filter(Boolean);
}

async function runClassification() {
  if (busy || !files.length) return;
  if (mode === "default" && !selectedCategoryIds().length) {
    $("runStatus").textContent = "분류할 기본 카테고리를 하나 이상 선택하세요.";
    return;
  }
  const queries = parseQueries();
  if (mode === "search" && !queries.length) {
    $("runStatus").textContent = "찾고 싶은 단어나 문장을 입력하세요.";
    return;
  }
  if (
    mode === "search" &&
    modelState.text.status !== "ready" &&
    !modelState.text.cached &&
    !textConsentGranted
  ) {
    updateSearchConsent();
    $("runStatus").textContent = "자유 검색 모델 추가 다운로드에 먼저 동의해 주세요.";
    $("searchConsent").scrollIntoView({ behavior: "smooth", block: "center" });
    return;
  }

  busy = true;
  refreshControls();
  $("results").innerHTML = "";
  try {
    await ensureImageEmbeddings();
    const threshold = Number($("threshold").value);
    let results;
    let labels;
    if (mode === "default") {
      const ids = selectedCategoryIds();
      results = await callWorker("classify-default", {
        ids: files.map((item) => item.id),
        selectedCategoryIds: ids,
        threshold,
      });
      labels = new Map(ids.map((id) => [id, categoryById.get(id)?.labelKo ?? id]));
    } else {
      await prepareModel("text");
      results = await callWorker("search-images", {
        ids: files.map((item) => item.id),
        queries,
        threshold,
      });
      labels = new Map(queries.map((query, index) => [`query-${index}`, query]));
    }
    renderResults(results, labels);
    $("runStatus").textContent = `완료 — ${files.length}장 분석 · ${mode === "default" ? "기본 자동분류" : "자유 검색"}`;
  } catch (error) {
    console.error(error);
    $("runStatus").textContent = `⚠️ ${error.message} · 모델 상태에서 다시 시도할 수 있습니다.`;
  } finally {
    busy = false;
    refreshControls();
  }
}

function renderResults(results, labels) {
  const buckets = new Map([...labels].map(([id, label]) => [id, { label, items: [] }]));
  const unsorted = [];
  const fileById = new Map(files.map((item) => [item.id, item]));
  for (const result of results) {
    if (!result.matches.length) {
      const item = {
        ...fileById.get(result.id),
        score: result.top3[0]?.score ?? 0,
        top3: result.top3,
      };
      unsorted.push(item);
      continue;
    }
    for (const match of result.matches) {
      if (!buckets.has(match.id)) continue;
      buckets.get(match.id).items.push({
        ...fileById.get(result.id),
        score: match.score,
        similarity: match.similarity,
        margin: match.margin,
        top3: result.top3,
      });
    }
  }

  const root = $("results");
  root.innerHTML = "";
  for (const { label, items } of buckets.values()) {
    root.appendChild(renderGroup(`📁 ${label}`, label, items, true));
  }
  root.appendChild(renderGroup("🗂️ 미분류", "기타", unsorted, false));
}

function renderGroup(title, zipName, items, allowZip) {
  const wrapper = document.createElement("section");
  wrapper.className = "group card";
  const header = document.createElement("div");
  header.className = "group-header";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const count = document.createElement("span");
  count.className = "count";
  count.textContent = `${items.length}장`;
  heading.appendChild(count);
  header.appendChild(heading);
  if (allowZip && items.length) {
    const download = document.createElement("button");
    download.className = "ghost small";
    download.textContent = "⬇ ZIP으로 저장";
    download.addEventListener("click", () => downloadZip(zipName, items));
    header.appendChild(download);
  }
  wrapper.appendChild(header);

  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "muted";
    empty.textContent = "해당하는 사진이 없습니다.";
    wrapper.appendChild(empty);
    return wrapper;
  }

  const grid = document.createElement("div");
  grid.className = "grid";
  items.sort((left, right) => right.score - left.score);
  for (const item of items) {
    const thumb = document.createElement("figure");
    thumb.className = "thumb";
    const image = document.createElement("img");
    image.src = item.url;
    image.alt = item.file.name;
    image.loading = "lazy";
    const score = document.createElement("figcaption");
    score.className = "score";
    score.textContent = `${(item.score * 100).toFixed(0)}%`;
    score.dataset.similarity = String(item.similarity ?? "");
    score.dataset.margin = String(item.margin ?? "");
    score.title = item.top3
      .map((entry) => `${entry.label} ${(entry.score * 100).toFixed(1)}%`)
      .join("\n");
    thumb.append(image, score);
    grid.appendChild(thumb);
  }
  wrapper.appendChild(grid);
  return wrapper;
}

async function resetPhotos() {
  for (const item of files) URL.revokeObjectURL(item.url);
  files = [];
  embeddedIds.clear();
  await callWorker("clear-embeddings");
  $("fileInput").value = "";
  $("results").innerHTML = "";
  $("runStatus").textContent = "";
  renderSelectedFiles();
  refreshControls();
}

async function clearAiCache() {
  if (!confirm("AI 모델 캐시만 삭제할까요? 선택한 사진과 현재 결과는 삭제하지 않습니다.")) return;
  $("cacheClearBtn").disabled = true;
  try {
    await callWorker("unload-models", { kind: "all" });
    embeddedIds.clear();
    await clearModelCache("all");
    for (const kind of ["vision", "text"]) {
      modelState[kind] = {
        status: "idle",
        progress: 0,
        loaded: 0,
        error: null,
        cached: false,
      };
      modelLoadPromises[kind] = null;
      renderModelState(kind);
    }
    $("runStatus").textContent = "AI 모델 캐시를 삭제했습니다. 사진 목록은 유지됩니다.";
    await refreshCacheStatus();
  } finally {
    $("cacheClearBtn").disabled = false;
    refreshControls();
  }
}

function refreshControls() {
  const runButton = $("runBtn");
  runButton.disabled = busy || !files.length;
  runButton.textContent = busy
    ? "처리 중…"
    : mode === "default"
      ? "빠른 자동분류"
      : "자유 검색";
  $("resetBtn").disabled = busy || !files.length;
  updateSearchConsent();
}

function bindEvents() {
  for (const button of document.querySelectorAll(".mode-button")) {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  }

  $("visionAction").addEventListener("click", () =>
    prepareModel("vision").catch((error) => console.error(error)),
  );
  $("textAction").addEventListener("click", () => {
    textConsentGranted = true;
    updateSearchConsent();
    prepareModel("text").catch((error) => console.error(error));
  });
  $("confirmTextDownload").addEventListener("click", () => {
    textConsentGranted = true;
    updateSearchConsent();
    prepareModel("text").catch((error) => console.error(error));
  });
  $("cacheClearBtn").addEventListener("click", clearAiCache);
  $("runBtn").addEventListener("click", runClassification);
  $("resetBtn").addEventListener("click", resetPhotos);
  $("restartWorkerBtn").addEventListener("click", async () => {
    $("restartWorkerBtn").classList.add("hidden");
    embeddedIds.clear();
    await createWorker();
    $("runStatus").textContent = "AI 작업 공간을 다시 시작했습니다. 다시 실행해 주세요.";
    refreshControls();
  });

  const dropzone = $("dropzone");
  dropzone.addEventListener("click", () => $("fileInput").click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") $("fileInput").click();
  });
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("drag");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag");
    addFiles(event.dataTransfer.files);
  });
  $("fileInput").addEventListener("change", (event) => addFiles(event.target.files));
  $("threshold").addEventListener("input", (event) => {
    $("thresholdValue").textContent = Number(event.target.value).toFixed(2);
  });
  $("selectAllCategories").addEventListener("click", () => {
    document.querySelectorAll("#categoryChips input").forEach((input) => {
      input.checked = true;
    });
    refreshControls();
  });
  $("clearCategories").addEventListener("click", () => {
    document.querySelectorAll("#categoryChips input").forEach((input) => {
      input.checked = false;
    });
    refreshControls();
  });
}

async function init() {
  renderCategories();
  bindEvents();
  renderSelectedFiles();
  renderModelState("vision");
  renderModelState("text");
  setMode("default");
  await createWorker();
  await refreshCacheStatus();
  $("runtimeInfo").textContent =
    `Transformers.js ${TRANSFORMERS_VERSION} · ${MODEL_CONFIG.dtype} · CPU/WASM · ` +
    `${MODEL_CONFIG.modelVersion}`;

  window.__PHOTO_SORTER_DEBUG__ = {
    getState: () => ({
      mode,
      fileCount: files.length,
      embeddedCount: embeddedIds.size,
      modelState: structuredClone(modelState),
    }),
  };
}

init().catch((error) => {
  console.error(error);
  $("runStatus").textContent = `초기화 실패: ${error.message}`;
});

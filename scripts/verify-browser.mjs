import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium } from "playwright-core";
import sharp from "sharp";

const APP_URL = process.env.APP_URL ?? "http://127.0.0.1:4173/";
const CHROME_PATH =
  process.env.CHROME_PATH ?? "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe";
const EDGE_PATH =
  process.env.EDGE_PATH ?? "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const REPORT_DIR = resolve("reports");
const SCREENSHOT_DIR = join(REPORT_DIR, "screenshots");

await mkdir(SCREENSHOT_DIR, { recursive: true });

const fixtureDir = await mkdtemp(join(tmpdir(), "photo-sorter-fixtures-"));
const fixtures = [
  {
    name: "cats.jpg",
    url: "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg",
  },
  {
    name: "football-match.jpg",
    url: "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg",
  },
];
for (const fixture of fixtures) {
  const response = await fetch(fixture.url);
  if (!response.ok) throw new Error(`테스트 이미지 다운로드 실패: ${fixture.url}`);
  fixture.path = join(fixtureDir, fixture.name);
  await writeFile(fixture.path, Buffer.from(await response.arrayBuffer()));
}
const personAnimalFixture = {
  name: "person-and-animal.jpg",
  path: join(fixtureDir, "person-and-animal.jpg"),
};
const [catsHalf, peopleHalf] = await Promise.all([
  sharp(fixtures[0].path).resize(500, 500, { fit: "cover" }).jpeg().toBuffer(),
  sharp(fixtures[1].path).resize(500, 500, { fit: "cover" }).jpeg().toBuffer(),
]);
await sharp({
  create: {
    width: 1000,
    height: 500,
    channels: 3,
    background: "#ffffff",
  },
})
  .composite([
    { input: catsHalf, left: 0, top: 0 },
    { input: peopleHalf, left: 500, top: 0 },
  ])
  .jpeg()
  .toFile(personAnimalFixture.path);
fixtures.push(personAnimalFixture);

const report = {
  testedAt: new Date().toISOString(),
  appUrl: APP_URL,
  chrome: {},
  edge: {},
  failures: [],
};
const openedContexts = [];

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function isModelRequest(url) {
  return /\/onnx\/(?:model|vision_model|text_model).+\.onnx/.test(url);
}

function collectNetwork(page) {
  const requests = [];
  const responses = [];
  page.on("request", (request) => {
    requests.push({
      url: request.url(),
      method: request.method(),
      resourceType: request.resourceType(),
      at: Date.now(),
    });
  });
  page.on("response", async (response) => {
    const url = response.url();
    if (!isModelRequest(url)) return;
    const headers = await response.allHeaders();
    responses.push({
      url,
      status: response.status(),
      contentLength: Number(headers["content-length"] ?? 0),
      fromServiceWorker: response.fromServiceWorker(),
    });
  });
  return { requests, responses };
}

async function waitForApp(page) {
  await page.goto(APP_URL, { waitUntil: "domcontentloaded", timeout: 30_000 });
  await page.waitForFunction(() => Boolean(window.__PHOTO_SORTER_DEBUG__), null, {
    timeout: 30_000,
  });
}

async function waitForModel(page, kind, timeout = 480_000) {
  await page.waitForFunction(
    (modelKind) => {
      const state = window.__PHOTO_SORTER_DEBUG__?.getState().modelState[modelKind];
      return state?.status === "ready" || state?.status === "error";
    },
    kind,
    { timeout },
  );
  const state = await page.evaluate(
    (modelKind) => window.__PHOTO_SORTER_DEBUG__.getState().modelState[modelKind],
    kind,
  );
  if (state.status === "error") throw new Error(`${kind} 모델 실패: ${state.error}`);
  return state;
}

async function waitForRunComplete(page, timeout = 360_000) {
  await page.waitForFunction(
    () => document.querySelector("#runStatus")?.textContent?.includes("완료"),
    null,
    { timeout },
  );
}

async function fullChromeVerification() {
  const profileDir = await mkdtemp(join(tmpdir(), "photo-sorter-chrome-"));
  report.chrome.profileDir = profileDir;

  let context = await chromium.launchPersistentContext(profileDir, {
    executablePath: CHROME_PATH,
    headless: true,
    viewport: { width: 1280, height: 900 },
  });
  openedContexts.push(context);
  let page = context.pages()[0] ?? (await context.newPage());
  let network = collectNetwork(page);
  const consoleErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });

  await waitForApp(page);
  await page.waitForTimeout(2_000);
  const initialModelRequests = network.requests.filter(({ url }) => isModelRequest(url));
  assert(initialModelRequests.length === 0, "첫 방문에서 모델 요청이 발생했습니다.");

  await page.setInputFiles("#fileInput", fixtures.map((fixture) => fixture.path));
  await page.waitForFunction(
    () => window.__PHOTO_SORTER_DEBUG__.getState().fileCount === 3,
    null,
    { timeout: 10_000 },
  );
  await page.waitForTimeout(1_000);
  assert(
    network.requests.filter(({ url }) => isModelRequest(url)).length === 0,
    "사진 선택만으로 모델 요청이 발생했습니다.",
  );

  await page.click("#runBtn");
  await waitForModel(page, "vision");
  await waitForRunComplete(page);
  const afterDefault = network.requests.filter(({ url }) => isModelRequest(url));
  assert(
    afterDefault.some(({ url }) => url.includes("vision_model_quantized.onnx")),
    "자동분류에서 이미지 모델 요청이 확인되지 않았습니다.",
  );
  assert(
    !afterDefault.some(({ url }) => url.includes("text_model_quantized.onnx")),
    "자동분류 중 텍스트 모델 요청이 발생했습니다.",
  );
  const defaultGroups = await page.locator("#results .group").count();
  assert(defaultGroups >= 2, "기본 자동분류 결과 그룹이 렌더링되지 않았습니다.");
  const defaultAssignments = await page.$$eval("#results .group", (groups) =>
    groups
      .map((group) => ({
        title: group.querySelector("h3")?.textContent?.trim() ?? "",
        images: [...group.querySelectorAll(".thumb img")].map((image) => image.alt),
      }))
      .filter((group) => group.images.length),
  );
  assert(
    defaultAssignments.some(
      (group) => group.title.includes("동물") && group.images.includes("cats.jpg"),
    ),
    "고양이 테스트 이미지가 동물 카테고리에 나타나지 않았습니다.",
  );
  assert(
    defaultAssignments.some(
      (group) =>
        group.title.includes("사람") && group.images.includes("person-and-animal.jpg"),
    ),
    "사람+동물 테스트 이미지가 사람 카테고리에 나타나지 않았습니다.",
  );
  assert(
    defaultAssignments.some(
      (group) =>
        group.title.includes("동물") && group.images.includes("person-and-animal.jpg"),
    ),
    "사람+동물 테스트 이미지가 동물 카테고리에 나타나지 않았습니다.",
  );

  const textRequestsBeforeMode = afterDefault.filter(({ url }) =>
    url.includes("text_model_quantized.onnx"),
  ).length;
  await page.click('[data-mode="search"]');
  await page.click("#runBtn");
  await page.waitForFunction(
    () => document.querySelector("#runStatus")?.textContent?.includes("동의"),
    null,
    { timeout: 10_000 },
  );
  assert(
    network.requests.filter(({ url }) => url.includes("text_model_quantized.onnx")).length ===
      textRequestsBeforeMode,
    "자유 검색 동의 전에 텍스트 모델 요청이 발생했습니다.",
  );

  await page.click("#confirmTextDownload");
  await waitForModel(page, "text");
  assert(
    network.requests.some(({ url }) => url.includes("text_model_quantized.onnx")),
    "동의 후 텍스트 모델 요청이 확인되지 않았습니다.",
  );
  await page.fill("#searchInput", "cat, football match");
  await page.click("#runBtn");
  await waitForRunComplete(page);
  const searchGroups = await page.locator("#results .group").count();
  assert(searchGroups >= 3, "자유 검색 결과 그룹이 렌더링되지 않았습니다.");
  const searchAssignments = await page.$$eval("#results .group", (groups) =>
    groups
      .map((group) => ({
        title: group.querySelector("h3")?.textContent?.trim() ?? "",
        images: [...group.querySelectorAll(".thumb img")].map((image) => image.alt),
      }))
      .filter((group) => group.images.length),
  );
  const zipButtons = page.locator("#results .group button");
  assert((await zipButtons.count()) > 0, "ZIP 저장 버튼이 생성되지 않았습니다.");
  const downloadPromise = page.waitForEvent("download");
  await zipButtons.first().click();
  const download = await downloadPromise;
  const zipFilename = download.suggestedFilename();
  assert(zipFilename.endsWith(".zip"), "ZIP 다운로드 파일명이 올바르지 않습니다.");

  const cacheKeys = await page.evaluate(async () => {
    const cache = await caches.open("transformers-cache");
    return (await cache.keys()).map((request) => request.url);
  });
  assert(
    cacheKeys.some((url) => url.includes("vision_model_quantized.onnx")),
    "이미지 모델이 Cache Storage에 저장되지 않았습니다.",
  );
  assert(
    cacheKeys.some((url) => url.includes("text_model_quantized.onnx")),
    "텍스트 모델이 Cache Storage에 저장되지 않았습니다.",
  );
  await page.screenshot({
    path: join(SCREENSHOT_DIR, "chrome-first-visit.png"),
    fullPage: true,
  });

  report.chrome.firstVisit = {
    initialModelRequestCount: initialModelRequests.length,
    modelRequestsAfterFileSelection: 0,
    modelRequests: network.requests.filter(({ url }) => isModelRequest(url)),
    modelResponses: network.responses,
    defaultGroups,
    defaultAssignments,
    searchGroups,
    searchAssignments,
    zipFilename,
    cacheModelFiles: cacheKeys.filter((url) => isModelRequest(url)),
    consoleErrors,
  };
  await page.click("#resetBtn");
  await page.waitForFunction(
    () => window.__PHOTO_SORTER_DEBUG__.getState().fileCount === 0,
    null,
    { timeout: 10_000 },
  );
  report.chrome.firstVisit.resetPreservedModels = await page.evaluate(() => {
    const state = window.__PHOTO_SORTER_DEBUG__.getState();
    return state.modelState.vision.status === "ready" && state.modelState.text.status === "ready";
  });
  await context.close();

  context = await chromium.launchPersistentContext(profileDir, {
    executablePath: CHROME_PATH,
    headless: true,
    viewport: { width: 1280, height: 900 },
  });
  openedContexts.push(context);
  page = context.pages()[0] ?? (await context.newPage());
  network = collectNetwork(page);
  let blockedHuggingFaceRequests = 0;
  await page.route("https://huggingface.co/**", async (route) => {
    blockedHuggingFaceRequests += 1;
    await route.abort();
  });
  await waitForApp(page);
  const initialState = await page.evaluate(() => window.__PHOTO_SORTER_DEBUG__.getState());
  assert(initialState.modelState.vision.cached, "재방문 시 이미지 모델 캐시가 감지되지 않았습니다.");
  assert(initialState.modelState.text.cached, "재방문 시 텍스트 모델 캐시가 감지되지 않았습니다.");
  await page.setInputFiles("#fileInput", fixtures.map((fixture) => fixture.path));
  await page.click("#runBtn");
  await waitForRunComplete(page);
  await page.click('[data-mode="search"]');
  await page.fill("#searchInput", "cat");
  await page.click("#runBtn");
  await waitForRunComplete(page);
  assert(blockedHuggingFaceRequests === 0, "재방문 캐시 사용 중 외부 모델 요청이 발생했습니다.");
  assert(
    network.requests.filter(({ url }) => isModelRequest(url)).length === 0,
    "재방문에서 모델 네트워크 요청이 관찰되었습니다.",
  );
  await page.screenshot({
    path: join(SCREENSHOT_DIR, "chrome-revisit-cache.png"),
    fullPage: true,
  });
  report.chrome.revisit = {
    cachedVision: initialState.modelState.vision.cached,
    cachedText: initialState.modelState.text.cached,
    externalModelRequests: blockedHuggingFaceRequests,
    observedModelRequests: network.requests.filter(({ url }) => isModelRequest(url)).length,
  };
  await context.close();

  context = await chromium.launchPersistentContext(profileDir, {
    executablePath: CHROME_PATH,
    headless: true,
    viewport: { width: 1280, height: 900 },
  });
  openedContexts.push(context);
  page = context.pages()[0] ?? (await context.newPage());
  await page.goto(`${APP_URL}?testModelError=vision`, {
    waitUntil: "domcontentloaded",
    timeout: 30_000,
  });
  await page.waitForFunction(() => Boolean(window.__PHOTO_SORTER_DEBUG__), null, {
    timeout: 30_000,
  });
  await page.click("#visionAction");
  await page.waitForFunction(
    () => window.__PHOTO_SORTER_DEBUG__.getState().modelState.vision.status === "error",
    null,
    { timeout: 20_000 },
  );
  const errorText = await page.locator("#visionStatus").innerText();
  await page.click("#visionAction");
  await waitForModel(page, "vision");
  const retryState = await page.evaluate(
    () => window.__PHOTO_SORTER_DEBUG__.getState().modelState.vision,
  );
  await page.setInputFiles("#fileInput", fixtures[0].path);
  page.once("dialog", (dialog) => dialog.accept());
  await page.click("#cacheClearBtn");
  await page.waitForFunction(
    () => !window.__PHOTO_SORTER_DEBUG__.getState().modelState.vision.cached,
    null,
    { timeout: 30_000 },
  );
  const stateAfterCacheClear = await page.evaluate(() => window.__PHOTO_SORTER_DEBUG__.getState());
  assert(stateAfterCacheClear.fileCount === 1, "모델 캐시 삭제가 사진 목록까지 삭제했습니다.");
  report.chrome.errorRecovery = {
    injectedErrorText: errorText,
    retrySucceeded: retryState.status === "ready",
    photoCountAfterCacheClear: stateAfterCacheClear.fileCount,
    cacheCleared: !stateAfterCacheClear.modelState.vision.cached,
  };
  await context.close();
}

async function edgeSmokeVerification() {
  const profileDir = await mkdtemp(join(tmpdir(), "photo-sorter-edge-"));
  const context = await chromium.launchPersistentContext(profileDir, {
    executablePath: EDGE_PATH,
    headless: true,
    viewport: { width: 1280, height: 900 },
  });
  openedContexts.push(context);
  const page = context.pages()[0] ?? (await context.newPage());
  const network = collectNetwork(page);
  await waitForApp(page);
  await page.waitForTimeout(2_000);
  assert(
    network.requests.filter(({ url }) => isModelRequest(url)).length === 0,
    "Edge 첫 방문에서 모델 요청이 발생했습니다.",
  );
  await page.click('[data-mode="search"]');
  await page.waitForTimeout(500);
  assert(await page.locator("#searchConsent").isVisible(), "Edge에서 자유 검색 동의 UI가 보이지 않습니다.");
  assert(
    network.requests.filter(({ url }) => isModelRequest(url)).length === 0,
    "Edge에서 자유 검색 모드 선택만으로 모델 요청이 발생했습니다.",
  );
  await page.screenshot({
    path: join(SCREENSHOT_DIR, "edge-first-visit.png"),
    fullPage: true,
  });
  report.edge = {
    initialModelRequestCount: 0,
    searchConsentVisible: true,
    modeSelectionModelRequestCount: 0,
  };
  await context.close();
}

try {
  console.log("Chrome 전체 검증 시작");
  await fullChromeVerification();
  console.log("Chrome 전체 검증 통과");
} catch (error) {
  report.failures.push({ browser: "Chrome", message: error.message, stack: error.stack });
  console.error(error);
}

try {
  console.log("Edge 스모크 검증 시작");
  await edgeSmokeVerification();
  console.log("Edge 스모크 검증 통과");
} catch (error) {
  report.failures.push({ browser: "Edge", message: error.message, stack: error.stack });
  console.error(error);
}

for (const context of openedContexts) {
  await context.close().catch(() => {});
}

await writeFile(
  join(REPORT_DIR, "browser-test-results.json"),
  `${JSON.stringify(report, null, 2)}\n`,
  "utf8",
);

if (report.failures.length) {
  process.exitCode = 1;
} else {
  console.log("모든 브라우저 검증 통과");
}

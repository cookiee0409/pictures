import { chromium } from "playwright-core";
import { createServer } from "vite";
import { writeFile } from "node:fs/promises";

const inputFiles =
  process.argv.length > 2
    ? process.argv.slice(2)
    : [
        "E:\\사진\\간사이여행\\IMG_3702.JPG",
        "E:\\사진\\간사이여행\\IMG_3717.JPG",
      ];
const isUserPhotoRegression = process.argv.length === 2;

const server = await createServer({
  server: { host: "127.0.0.1", port: 4173 },
  configLoader: "runner",
});
await server.listen();

const context = await chromium.launchPersistentContext(
  "C:\\Users\\USER\\AppData\\Local\\Temp\\photo-sorter-chrome-pS2UHj",
  {
    executablePath: "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    headless: true,
  },
);

try {
  const page = context.pages()[0] ?? (await context.newPage());
  await page.goto("http://127.0.0.1:4173/", { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => Boolean(window.__PHOTO_SORTER_DEBUG__));
  await page.setInputFiles("#fileInput", inputFiles);
  await page.click("#runBtn");
  await page.waitForFunction(
    () => document.querySelector("#runStatus")?.textContent?.includes("완료"),
    null,
    { timeout: 360_000 },
  );
  const payload = await page.evaluate(() => ({
    results: window.__PHOTO_SORTER_DEBUG__.getLastResults(),
    groups: [...document.querySelectorAll("#results .group")].map((group) => ({
      title: group.querySelector("h3")?.textContent,
      images: [...group.querySelectorAll("img")].map((image) => image.alt),
    })),
  }));
  if (isUserPhotoRegression) {
    await writeFile(
      new URL("../reports/user-photo-regression.json", import.meta.url),
      `${JSON.stringify(
        {
          testedAt: new Date().toISOString(),
          browser: "Google Chrome (Playwright Core)",
          privacy: "The source photos and image embeddings stayed in the local browser process.",
          ...payload,
        },
        null,
        2,
      )}\n`,
      "utf8",
    );
  }
  console.log(JSON.stringify(payload, null, 2));
} finally {
  await context.close();
  await server.close();
}

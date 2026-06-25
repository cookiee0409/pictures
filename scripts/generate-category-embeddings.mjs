import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import {
  AutoTokenizer,
  CLIPTextModelWithProjection,
  env,
} from "@huggingface/transformers";
import { DEFAULT_CATEGORIES } from "../src/config/categories.js";
import { MODEL_CONFIG, TRANSFORMERS_VERSION } from "../src/config/model-config.js";
import { averageEmbeddings, l2Normalize } from "../src/ai/similarity.js";

env.allowLocalModels = false;
env.useBrowserCache = false;

const outputPath = resolve("public/data/category-embeddings.json");
let lastProgress = -1;

console.log(`텍스트 인코더 로드: ${MODEL_CONFIG.id}@${MODEL_CONFIG.revision}`);
const [tokenizer, model] = await Promise.all([
  AutoTokenizer.from_pretrained(MODEL_CONFIG.id, {
    revision: MODEL_CONFIG.revision,
  }),
  CLIPTextModelWithProjection.from_pretrained(MODEL_CONFIG.id, {
    revision: MODEL_CONFIG.revision,
    dtype: MODEL_CONFIG.dtype,
    device: "cpu",
    progress_callback: (event) => {
      if (event.status === "progress" && event.file.endsWith(".onnx")) {
        const progress = Math.floor(event.progress);
        if (progress === lastProgress && progress !== 100) return;
        lastProgress = progress;
        process.stdout.write(
          `\r${event.file}: ${event.progress.toFixed(1)}% (${(event.loaded / 1e6).toFixed(1)}MB)`,
        );
      }
    },
  }),
]);
process.stdout.write("\n");

const categories = [];
for (const category of DEFAULT_CATEGORIES) {
  const inputs = tokenizer(category.prompts, { padding: true, truncation: true });
  const { text_embeds: textEmbeds } = await model(inputs);
  const promptEmbeddings = [];
  for (let row = 0; row < textEmbeds.dims[0]; row += 1) {
    const start = row * MODEL_CONFIG.embeddingDimension;
    promptEmbeddings.push(
      l2Normalize(textEmbeds.data.slice(start, start + MODEL_CONFIG.embeddingDimension)),
    );
  }
  const embedding = averageEmbeddings(promptEmbeddings);
  categories.push({
    id: category.id,
    labelKo: category.labelKo,
    labelEn: category.labelEn,
    prompts: category.prompts,
    hidden: Boolean(category.hidden),
    embedding: Array.from(embedding, (value) => Number(value.toFixed(8))),
  });
  console.log(`생성 완료: ${category.labelKo}`);
}

await model.dispose();

const payload = {
  schemaVersion: 1,
  model: MODEL_CONFIG.id,
  revision: MODEL_CONFIG.revision,
  modelVersion: MODEL_CONFIG.modelVersion,
  transformersVersion: TRANSFORMERS_VERSION,
  dtype: MODEL_CONFIG.dtype,
  sourceModelFile: MODEL_CONFIG.files.text.path,
  embeddingDimension: MODEL_CONFIG.embeddingDimension,
  normalized: true,
  aggregation: "L2-normalized prompt mean, then L2-normalized",
  generatedAt: new Date().toISOString(),
  categories,
};

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(payload)}\n`, "utf8");
console.log(`저장 완료: ${outputPath}`);

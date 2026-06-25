import { MODEL_CONFIG } from "../config/model-config.js";

let categoryDataPromise;

export async function loadCategoryEmbeddings(url) {
  categoryDataPromise ??= fetch(url, { cache: "force-cache" }).then(async (response) => {
    if (!response.ok) {
      throw new Error(`카테고리 임베딩을 불러오지 못했습니다. (${response.status})`);
    }
    const data = await response.json();
    validateCategoryEmbeddings(data);
    return {
      ...data,
      categories: data.categories.map((category) => ({
        ...category,
        embedding: Float32Array.from(category.embedding),
        negativeEmbedding: Float32Array.from(category.negativeEmbedding),
        promptEmbeddings: category.promptEmbeddings.map((values) => Float32Array.from(values)),
        negativePromptEmbeddings: category.negativePromptEmbeddings.map((values) =>
          Float32Array.from(values),
        ),
      })),
    };
  });
  return categoryDataPromise;
}
export function validateCategoryEmbeddings(data) {
  if (data.model !== MODEL_CONFIG.id) {
    throw new Error(`카테고리 임베딩 모델 불일치: ${data.model}`);
  }
  if (data.revision !== MODEL_CONFIG.revision) {
    throw new Error("카테고리 임베딩 revision이 현재 모델과 다릅니다.");
  }
  if (data.modelVersion !== MODEL_CONFIG.modelVersion) {
    throw new Error("카테고리 임베딩 데이터 버전이 현재 앱과 다릅니다.");
  }
  if (data.embeddingDimension !== MODEL_CONFIG.embeddingDimension || !data.normalized) {
    throw new Error("카테고리 임베딩 차원 또는 정규화 정보가 올바르지 않습니다.");
  }
  if (!Array.isArray(data.categories) || !data.categories.length) {
    throw new Error("카테고리 임베딩이 비어 있습니다.");
  }
  for (const category of data.categories) {
    if (category.embedding?.length !== MODEL_CONFIG.embeddingDimension) {
      throw new Error(`${category.id} 카테고리 임베딩 차원이 올바르지 않습니다.`);
    }
    if (category.negativeEmbedding?.length !== MODEL_CONFIG.embeddingDimension) {
      throw new Error(`${category.id} 부정 임베딩 차원이 올바르지 않습니다.`);
    }
    if (
      !category.promptEmbeddings?.length ||
      category.promptEmbeddings.some(
        (embedding) => embedding.length !== MODEL_CONFIG.embeddingDimension,
      )
    ) {
      throw new Error(`${category.id} 프롬프트 임베딩이 올바르지 않습니다.`);
    }
    if (
      !category.negativePromptEmbeddings?.length ||
      category.negativePromptEmbeddings.some(
        (embedding) => embedding.length !== MODEL_CONFIG.embeddingDimension,
      )
    ) {
      throw new Error(`${category.id} 부정 프롬프트 임베딩이 올바르지 않습니다.`);
    }
  }
}

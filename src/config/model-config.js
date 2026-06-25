export const TRANSFORMERS_VERSION = "3.8.1";
export const CATEGORY_EMBEDDINGS_VERSION = "independent-multilabel-v2";

export const MODEL_CONFIG = Object.freeze({
  id: "Xenova/clip-vit-base-patch32",
  revision: "d15189d7028b43f1d3e65039190477f6af591c2a",
  modelVersion: "clip-vit-base-patch32-q8-d15189d",
  dtype: "q8",
  device: "wasm",
  embeddingDimension: 512,
  cacheName: "transformers-cache",
  remoteHost: "https://huggingface.co",
  remotePathTemplate: "{model}/resolve/{revision}",
  files: {
    integrated: {
      path: "onnx/model_quantized.onnx",
      bytes: 153_695_702,
    },
    vision: {
      path: "onnx/vision_model_quantized.onnx",
      bytes: 89_117_001,
    },
    text: {
      path: "onnx/text_model_quantized.onnx",
      bytes: 64_504_507,
    },
  },
});

export const EXPERIMENTAL_MODEL_CONFIG = Object.freeze({
  q4VisionEnabled: false,
  webGpuEnabled: false,
  mobileClipEnabled: false,
});

export function modelFileUrl(path) {
  const remotePath = MODEL_CONFIG.remotePathTemplate
    .replace("{model}", MODEL_CONFIG.id)
    .replace("{revision}", encodeURIComponent(MODEL_CONFIG.revision));
  return `${MODEL_CONFIG.remoteHost}/${remotePath}/${path}`;
}

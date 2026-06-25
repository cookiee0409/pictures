import test from "node:test";
import assert from "node:assert/strict";
import {
  averageEmbeddings,
  cosineSimilarity,
  l2Normalize,
  rankCandidates,
  softmax,
} from "../src/ai/similarity.js";

test("l2Normalize은 단위 벡터를 만든다", () => {
  const normalized = l2Normalize([3, 4]);
  assert.ok(Math.abs(normalized[0] - 0.6) < 1e-6);
  assert.ok(Math.abs(normalized[1] - 0.8) < 1e-6);
});
test("averageEmbeddings은 평균 후 다시 정규화한다", () => {
  const average = averageEmbeddings([
    new Float32Array([1, 0]),
    new Float32Array([0, 1]),
  ]);
  assert.ok(Math.abs(average[0] - Math.SQRT1_2) < 1e-6);
  assert.ok(Math.abs(average[1] - Math.SQRT1_2) < 1e-6);
});

test("cosineSimilarity은 같은 방향을 1로 계산한다", () => {
  assert.ok(Math.abs(cosineSimilarity([0.6, 0.8], [0.6, 0.8]) - 1) < 1e-6);
});

test("softmax와 rankCandidates는 가장 가까운 후보를 먼저 반환한다", () => {
  const probabilities = softmax([0.2, 0.1], 10);
  assert.ok(probabilities[0] > probabilities[1]);

  const ranked = rankCandidates(
    new Float32Array([1, 0]),
    [
      { id: "a", label: "A", embedding: new Float32Array([1, 0]) },
      { id: "b", label: "B", embedding: new Float32Array([0, 1]) },
    ],
    10,
  );
  assert.equal(ranked[0].id, "a");
  assert.ok(ranked[0].score > 0.99);
});

import test from "node:test";
import assert from "node:assert/strict";
import {
  averageEmbeddings,
  binarySimilarityScore,
  cosineSimilarity,
  l2Normalize,
  passesIndependentThreshold,
  rankCandidates,
  scoreCandidatesIndependently,
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

test("카테고리별 이진 점수는 다른 카테고리 수와 무관하다", () => {
  const image = new Float32Array([1, 0]);
  const positive = new Float32Array([1, 0]);
  const negative = new Float32Array([0, 1]);
  const result = binarySimilarityScore(image, positive, negative, 10);
  assert.ok(result.score > 0.99);
  assert.ok(result.margin > 0);
});

test("독립 점수에서는 여러 카테고리가 동시에 기준을 넘을 수 있다", () => {
  const image = l2Normalize([1, 1]);
  const candidates = [
    {
      id: "person",
      label: "사람",
      embedding: l2Normalize([1, 0.8]),
      negativeEmbedding: l2Normalize([-1, 0]),
    },
    {
      id: "animal",
      label: "동물",
      embedding: l2Normalize([0.8, 1]),
      negativeEmbedding: l2Normalize([0, -1]),
    },
  ];
  const matches = scoreCandidatesIndependently(image, candidates, 10).filter(
    ({ score }) => score >= 0.5,
  );
  assert.deepEqual(matches.map(({ id }) => id).sort(), ["animal", "person"]);
});

test("카테고리 보정 최소 점수는 사용자 기준보다 우선한다", () => {
  assert.equal(
    passesIndependentThreshold({ score: 0.64, minimumScore: 0.65 }, 0.5),
    false,
  );
  assert.equal(
    passesIndependentThreshold({ score: 0.81, minimumScore: 0.65 }, 0.5),
    true,
  );
  assert.equal(
    passesIndependentThreshold({ score: 0.81, minimumScore: 0.65 }, 0.85),
    false,
  );
});

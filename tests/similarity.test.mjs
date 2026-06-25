import test from "node:test";
import assert from "node:assert/strict";
import {
  averageEmbeddings,
  calibratedSimilarityScore,
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

test("보정 점수는 카테고리 유사도 기준에서 0.5가 된다", () => {
  assert.equal(calibratedSimilarityScore(0.24, 0.24), 0.5);
  assert.ok(calibratedSimilarityScore(0.28, 0.24) > 0.8);
  assert.ok(calibratedSimilarityScore(0.2, 0.24) < 0.2);
});

test("분할 이미지의 최대 증거로 여러 카테고리가 동시에 기준을 넘는다", () => {
  const imageViews = [l2Normalize([1, 0]), l2Normalize([0, 1])];
  const candidates = [
    {
      id: "person",
      label: "사람",
      similarityThreshold: 0.8,
      promptEmbeddings: [l2Normalize([1, 0])],
    },
    {
      id: "animal",
      label: "동물",
      similarityThreshold: 0.8,
      promptEmbeddings: [l2Normalize([0, 1])],
    },
  ];
  const matches = scoreCandidatesIndependently(imageViews, candidates).filter(
    ({ score }) => score >= 0.5,
  );
  assert.deepEqual(matches.map(({ id }) => id).sort(), ["animal", "person"]);
});

test("사용자 점수 기준만 최종 표시 여부를 결정한다", () => {
  assert.equal(passesIndependentThreshold({ score: 0.64 }, 0.5), true);
  assert.equal(passesIndependentThreshold({ score: 0.64 }, 0.7), false);
});

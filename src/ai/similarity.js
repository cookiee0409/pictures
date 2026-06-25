export function l2Normalize(values) {
  const output = values instanceof Float32Array ? new Float32Array(values) : Float32Array.from(values);
  let sum = 0;
  for (const value of output) sum += value * value;
  const norm = Math.sqrt(sum);
  if (!Number.isFinite(norm) || norm === 0) {
    throw new Error("0 또는 비정상 벡터는 정규화할 수 없습니다.");
  }
  for (let index = 0; index < output.length; index += 1) {
    output[index] /= norm;
  }
  return output;
}
export function cosineSimilarity(left, right) {
  if (left.length !== right.length) {
    throw new Error(`임베딩 차원이 다릅니다: ${left.length} != ${right.length}`);
  }
  let score = 0;
  for (let index = 0; index < left.length; index += 1) {
    score += left[index] * right[index];
  }
  return score;
}

export function averageEmbeddings(embeddings) {
  if (!embeddings.length) throw new Error("평균낼 임베딩이 없습니다.");
  const dimension = embeddings[0].length;
  const average = new Float32Array(dimension);
  for (const embedding of embeddings) {
    if (embedding.length !== dimension) throw new Error("임베딩 차원이 다릅니다.");
    for (let index = 0; index < dimension; index += 1) {
      average[index] += embedding[index];
    }
  }
  for (let index = 0; index < dimension; index += 1) {
    average[index] /= embeddings.length;
  }
  return l2Normalize(average);
}

export function softmax(values, temperature = 100) {
  if (!values.length) return [];
  const scaled = values.map((value) => value * temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((value) => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
}

export function binarySimilarityScore(
  imageEmbedding,
  positiveEmbedding,
  negativeEmbedding,
  temperature = 50,
) {
  const positiveSimilarity = cosineSimilarity(imageEmbedding, positiveEmbedding);
  const negativeSimilarity = cosineSimilarity(imageEmbedding, negativeEmbedding);
  const [score] = softmax([positiveSimilarity, negativeSimilarity], temperature);
  return {
    score,
    similarity: positiveSimilarity,
    negativeSimilarity,
    margin: positiveSimilarity - negativeSimilarity,
  };
}

function maximumSimilarity(imageEmbedding, embeddings) {
  let maximum = -Infinity;
  for (const embedding of embeddings) {
    maximum = Math.max(maximum, cosineSimilarity(imageEmbedding, embedding));
  }
  return maximum;
}

export function scoreCandidatesIndependently(imageEmbedding, candidates, temperature = 50) {
  return candidates
    .map((candidate) => {
      const positiveSimilarity = maximumSimilarity(
        imageEmbedding,
        candidate.promptEmbeddings ?? [candidate.embedding],
      );
      const negativeSimilarity = maximumSimilarity(
        imageEmbedding,
        candidate.negativePromptEmbeddings ?? [candidate.negativeEmbedding],
      );
      const [score] = softmax([positiveSimilarity, negativeSimilarity], temperature);
      return {
        id: candidate.id,
        label: candidate.label,
        minimumScore: candidate.minimumScore,
        score,
        similarity: positiveSimilarity,
        negativeSimilarity,
        margin: positiveSimilarity - negativeSimilarity,
      };
    })
    .sort((left, right) => right.score - left.score);
}

export function passesIndependentThreshold(candidate, userThreshold) {
  return candidate.score >= Math.max(userThreshold, candidate.minimumScore ?? 0.5);
}

export function rankCandidates(imageEmbedding, candidates, temperature = 100) {
  const similarities = candidates.map((candidate) =>
    cosineSimilarity(imageEmbedding, candidate.embedding),
  );
  const probabilities = softmax(similarities, temperature);
  return candidates
    .map((candidate, index) => ({
      id: candidate.id,
      label: candidate.label,
      similarity: similarities[index],
      score: probabilities[index],
    }))
    .sort((left, right) => right.score - left.score);
}

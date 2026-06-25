# 🖼️ AI 사진 분류기

사진 원본·썸네일·이미지 임베딩을 외부 분석 서버로 보내지 않고 브라우저 안에서 분류하는 정적 웹앱입니다.

이번 버전은 기존 통합 CLIP 모델을 이미지/텍스트 인코더로 분리했습니다.

- 홈페이지 접속: 대용량 모델 다운로드 없음
- 빠른 자동분류: 이미지 모델 q8만 사용
- 자유 검색: 사용자가 동의할 때 텍스트 모델 q8 추가 다운로드
- 기본 카테고리: 사전 계산된 114KB 임베딩 JSON 사용
- 실행 경로: CPU/WASM
- 비활성 실험 경로: WebGPU, q4, MobileCLIP

## 실행

Node.js 20.19 이상이 필요합니다.

```bash
npm install
npm run dev
```

브라우저에서 표시된 로컬 주소를 엽니다. ES module과 Web Worker를 사용하므로 `index.html` 더블클릭 실행은 지원하지 않습니다.

## 빌드와 배포

```bash
npm run build
npm run preview
```

생성되는 `dist/` 폴더를 Cloudflare Pages, GitHub Pages, Vercel 정적 배포 등에 올릴 수 있습니다. 서버 측 사진 처리 API는 필요하지 않습니다.

## 모델과 실제 파일 크기

모델 저장소는 `Xenova/clip-vit-base-patch32`, revision은
`d15189d7028b43f1d3e65039190477f6af591c2a`로 고정했습니다.

| 경로 | 파일 | 실제 크기 |
|---|---|---:|
| 기존 통합 모델 | `onnx/model_quantized.onnx` | 153,695,702 bytes (153.7MB) |
| 빠른 자동분류 | `onnx/vision_model_quantized.onnx` | 89,117,001 bytes (89.1MB) |
| 자유 검색 추가 | `onnx/text_model_quantized.onnx` | 64,504,507 bytes (64.5MB) |

기본 자동분류의 모델 다운로드는 기존 대비 64,578,701 bytes, 약 42.0% 감소했습니다.

프로덕션 번들에는 ONNX Runtime WASM 파일도 포함됩니다. 현재 빌드 결과는 원본 21.6MB, gzip 약 5.1MB이며 모델을 실제로 준비할 때 필요합니다.

## 동작 구조

```text
페이지 접속
  └─ UI + Worker 코드 + 카테고리 임베딩만 준비

빠른 자동분류 실행
  ├─ CLIPVisionModelWithProjection q8 로드
  ├─ 사진 → 정규화된 image_embeds
  └─ 사전 계산된 카테고리 임베딩과 코사인 유사도 비교

자유 검색 동의
  ├─ CLIPTextModelWithProjection q8 추가 로드
  ├─ 검색 문장 → 정규화된 text_embeds
  └─ 메모리에 있는 이미지 임베딩과 비교
```

AI 추론, 이미지 전처리, L2 정규화와 유사도 계산은 `src/workers/ai-worker.js`에서 수행합니다.

## 카테고리 임베딩 다시 생성

기본 카테고리나 프롬프트를 바꾼 뒤 다음 명령을 실행합니다.

```bash
npm run generate:embeddings
```

이 명령은 고정된 q8 텍스트 인코더로 `public/data/category-embeddings.json`을 다시 만듭니다. 모델 ID, revision, Transformers.js 버전, 임베딩 차원과 정규화 방식을 파일에 함께 기록합니다.

## 캐시

- Transformers.js의 `transformers-cache` Cache Storage를 사용합니다.
- 모델 준비를 사용자가 시작할 때 `navigator.storage.persist()`를 요청합니다.
- 모델별 설치 상태를 화면에 표시합니다.
- `AI 모델 캐시 삭제`는 모델 캐시만 지우며 선택한 사진 목록은 유지합니다.
- 사진 초기화는 사진·결과·메모리 임베딩만 지우며 모델 캐시는 유지합니다.

## 테스트

```bash
npm test
npm run build
npm run verify:browser
```

2026-06-25 실제 Chrome 전체 시나리오와 Edge 첫 방문 스모크 테스트가 통과했습니다. 상세 결과는
[`reports/browser-test-results.json`](reports/browser-test-results.json)과
[`docs/작업-보고서.md`](docs/작업-보고서.md)에 있습니다.

## 알려진 한계

- 현재 자동 정확도 검증은 공개 고양이/축구 이미지의 기능 스모크 수준입니다. 100~200장 기준 정식 정확도 벤치마크는 아직 필요합니다.
- Transformers.js의 `from_pretrained()`는 이 구현에서 안정적인 다운로드 중단 신호를 제공하지 않아, 다운로드 실패 후 재시도는 지원하지만 진행 중 강제 취소는 지원하지 않습니다.
- Chrome에서 전체 모델·캐시·오류 복구를 검증했고 Edge는 모델 다운로드 전 지연 로딩 UI까지만 검증했습니다.
- Firefox와 Safari는 이번 실행 환경에서 직접 검증하지 못했습니다.
- q4, WebGPU, MobileCLIP은 검증 전 기본 경로로 승격하지 않습니다.

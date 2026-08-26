<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# Efficient LLM Router

**예산 제약 아래에서 프롬프트마다 최적 언어 모델을 고르는 라우터** · 2026 오픈소스 개발자대회 SK텔레콤 지정과제 출품작

[![tests](https://github.com/yuJunhyk/ossp-2026-llm-router-challenge/actions/workflows/test.yml/badge.svg)](https://github.com/yuJunhyk/ossp-2026-llm-router-challenge/actions/workflows/test.yml)

> | 항목 | 값 |
> | --- | --- |
> | train 교차검증 (vpCV, `analysis/cv_gate.py` template 그룹화) | **0.6675** · fold 예산 초과 0 (가드 확장 전 0.6529 · 초과 2) |
> | 공개 Dev 880문항 | **0.6780** · 비용 비율 1.111 / 1.662 / 3.214 (한도 1.25 / 2.0 / 4.0) |
> | 직전 구성(R1.4) 대비 | 배정·점수 동일 · 가드 확장은 공개 Dev에서 무동작, train 교차검증의 예산 초과 fold 2 → 0 |
>
> - 재현 절차: [5분 만에 돌려보기](#5분-만에-돌려보기)
> - 설계 결정과 기각 이력: [docs/decisions.md](docs/decisions.md)
> - 버전 이력: [CHANGELOG.md](CHANGELOG.md)
> - 한계와 향후 방향: [Issues](https://github.com/yuJunhyk/ossp-2026-llm-router-challenge/issues)
>
> | 역할 | 담당 범위 |
> | --- | --- |
> | 검증 · 캘리브레이션 | 사전 등록 게이트 운영, β·margin 재도출, 예산 안전 검증, 저장소 운영 |
> | 특징 · 가드 설계 | 서술형 밀집 특징 27종, 컨텍스트 소진 가드 |

프롬프트 본문만 읽고 문항마다 어느 언어 모델에 맡길지 **호출 전에** 정하는 라우터입니다. 쉬운 문항은 싼 모델로 보내고 아낀 예산을 판단이 뒤집히는 문항에 몰아주어, 정해진 비용 한도 안에서 평균 품질을 최대로 끌어올립니다. 추론은 파이썬 표준 라이브러리만 쓰며, 학습 결과는 1.11 MB JSON 하나로 이미지에 실립니다.

어려운 지점은 배분 알고리즘이 아니라 **비용 예측이 틀린다**는 데 있습니다. 어느 등급이든 총비용이 한도를 넘으면 그 등급은 부분 감점이 아니라 통째로 0점입니다. 그래서 이 라우터는 예측이 빗나가는 쪽을 실제보다 비싸게 값매겨 손대지 않고, 그렇게 확보한 여유를 예측이 잘 맞는 자리에만 씁니다.

2026 오픈소스 개발자대회 SK텔레콤 지정과제 출품작이며, 이 저장소는 주최측 스타터를 fork한 구현 저장소입니다.

## 결과

공개 Dev 880문항 기준입니다. 비용 비율은 같은 배치를 전부 최저가 모델로 처리했을 때를 1로 둔 상대값이고, 최종 점수는 등급 점수에 가중치를 곱해 더합니다.

| 등급 | 점수 | 비용 비율 | 한도 | 가중치 | 판정 |
|---|---:|---:|---:|---:|---|
| Fast | 0.6409 | 1.111 | 1.25 | 0.4 | 통과 |
| Balanced | 0.6847 | 1.662 | 2.0 | 0.3 | 통과 |
| Premium | 0.7207 | 3.214 | 4.0 | 0.3 | 통과 |
| 최종 | **0.6780** | — | — | — | 전 등급 한도 내 |

전량 최저가 배정이 0.6193, 정답을 미리 아는 배정의 상한이 0.8034이므로 라우팅으로 벌 수 있는 폭의 **31.9%**를 회수한 위치입니다. 이 폭이 0.184뿐이라는 사실 자체가 이 과제의 성격을 말해줍니다 — 가장 후한 Premium 등급에서조차 정답을 아는 배정이 880문항 중 617개를 최저가 모델에 남겨둡니다.

공식 제약 컨테이너에서 세 등급 모두 통과했고 등급당 6.2~6.3초가 걸립니다. 한도는 등급당 90초입니다. 실행에 네트워크도 GPU도 필요하지 않습니다.

## 5분 만에 돌려보기

설치할 의존성이 없습니다. Python 3.9 이상이면 됩니다.

toy 자료로 세 등급의 선택 결과를 만들고 채점합니다.

```console
for tier in fast balanced premium; do
  PYTHONPATH=src python3 -m ossp_router.heuristic \
    --input data/toy/inputs.json \
    --tier "$tier" \
    --output "build/toy/$tier.json"
done

PYTHONPATH=src python3 -m ossp_router.cli self-check \
  --input data/toy/inputs.json \
  --outcomes data/toy/outcomes.json \
  --submissions build/toy \
  --report build/toy-report.json
```

제출 파일 이름은 `fast.json` · `balanced.json` · `premium.json`이어야 합니다. `self-check`가 그 이름으로 읽습니다.

위 표의 수치를 재현하려면 공개 Dev 자료를 먼저 준비합니다. 일부 원천 자료는 라이선스 조건 때문에 내려받아 결합해야 합니다.

```console
python3 -m venv .venv-data
.venv-data/bin/pip install -r data/sources/requirements-materialize-public-data.txt
.venv-data/bin/python tools/materialize_public_data.py
```

그다음 위 명령에서 입력을 `data/materialized/dev/inputs.json`으로, outcomes를 `data/dev/outcomes.json`으로 바꿔 실행하면 됩니다.

## 어떻게 동작하나

한 등급을 처리하는 흐름은 일곱 단계입니다. 각 단계는 그것을 그렇게 만든 실측과 함께 적었습니다.

| 단계 | 하는 일 | 근거 |
|---|---|---|
| 1 | 프롬프트에서 특징 8,259개 추출 — 길이·기호·형식 수치 40개와 서술형 밀집 특징 27개에, n-gram을 crc32로 8,192칸에 접은 희소 벡터 | 밀집 특징 27종은 팀 동료가 설계해 게이트를 통과한 기여. 문헌이 제안한 구조 난이도 특징 26종은 잔차 상관이 전부 \|r\| < 0.05라 채택하지 않음 |
| 2 | dual ridge(λ=10)로 모델 셋의 점수와 로그 비용 여섯 값 예측 | 특징 8,259개에 문항 1,760개라 dual 형태가 맞음. GBM·앙상블·임베딩을 같은 조건에서 붙였으나 전부 열세 |
| 3 | 중간 모델 점수를 최저가 모델 점수 + 상수 0.081로 대체 | 문항별 격차 예측이 참값과 상관 0.033의 노이즈라, 그리디 순위가 그 노이즈를 소비하면 역선택이 생김 |
| 4 | 7자리 이상 정수와 함께 소수·인수분해 어휘가 나오거나, 같은 줄에서 거듭제곱(`x**3`) 뒤에 `= 0`이 오는 다항식 방정식이면 추론 모델의 예측 점수를 중간 모델 수준으로 눌러 승급 유인을 차단 | train에는 비용이 최저가의 천 배에서 만 배에 이르는 컨텍스트 소진 문항이 26건 있고 프롬프트만으로는 예측되지 않음 — 규칙이 닿는 범위는 원천에서 막음. 다항식 규칙이 새로 잡는 train 6문항은 L/M/K 점수 평균 0 / 0 / 0.08에 think 비용 4.1 credits라 승급 이득이 없고, 이 확장 하나로 train 교차검증의 예산 초과 fold가 2개에서 0개가 됨 (R1.5) |
| 5 | 예측 비용에 비관 쐐기 `exp(β·σ)`를 곱함 — β는 세 등급 모두 1.0 | β를 0.5로 낮춘 후보들이 새로 뽑은 fold 검사에서 다섯 번 예산을 터뜨림. 반쪽 쐐기는 고비용 문항 두어 건이 한 배치에 몰리는 사고를 견디지 못함 |
| 6 | 예산 상한에 margin을 곱함 — 0.94 · 1.08 · 0.96, Premium은 문항 800개 이상 배치에서 1.08 | β=1.0이 계획 비용을 두 배 가까이 부풀리므로 Balanced는 지갑을 그만큼 되돌려 열고, 예산이 문항 수에 비례하는 규칙 구조에 맞춰 큰 배치에서만 Premium 지갑을 넓힘. 배치가 작아지면 자동으로 보수 쪽으로 접힘 |
| 7 | 승급 증분을 점수 이득 ÷ 비용 순으로 사들이는 볼록껍질 그리디 배낭 | 동률은 문항 위치가 아니라 예측값으로 끊어, 문항을 섞거나 ID를 바꿔도 배정이 변하지 않음 |

라우터는 문항 ID·입력 순서·split·과제명을 읽지 않습니다. 모델 선택 함수가 받는 것은 프롬프트 본문과 실행 등급뿐입니다. 싼 모델을 먼저 돌려보고 결과를 본 뒤 승급하는 방식은 규칙이 금지하며, 이 라우터에는 그런 경로가 없습니다.

## 왜 이렇게 정했나

세 원칙이 모든 결정을 관통합니다. 신호가 없으면 배우지 않고 버립니다 — 문항별 격차 예측은 참값과 상관 0.033의 노이즈라 상수로 대체했습니다. 꼬리는 예측이 아니라 정책으로 막습니다 — β를 0.5로 낮춘 후보들이 새로 뽑은 fold에서 다섯 번 예산을 터뜨린 실측이 β=1.0의 근거입니다. 여유는 취향이 아니라 필요조건입니다 — 공개 Dev 예산의 99.6%를 쓰던 주최측 baseline이 채점셋에서 0점 처리된 전례가 있고, 이 라우터는 +12~24%의 비용 이동을 견딥니다. 각 결정의 실측 근거와 기각된 대안 전체는 [docs/decisions.md](docs/decisions.md)에 있습니다.

## 저장소 구조

| 경로 | 내용 |
|---|---|
| `src/ossp_router/learned_*.py` | 제출 이미지에 실리는 라우터 런타임. 표준 라이브러리만 사용 |
| `src/ossp_router/resources/learned-router.v1.json` | 학습 결과 아티팩트 1.11 MB |
| `analysis/` | 학습 파이프라인 세 파일과 교차검증 게이트. numpy만 필요하며 이미지에는 들어가지 않음 |
| `tests/test_learned_*.py` · `test_feature_parity.py` · `test_image_packaging.py` · `test_runtime_edges.py` | 라우터 계약·특징 동등성·패키징 테스트 |
| `docs/architecture.md` · `decisions.md` | 설계와 결정 기록 |
| `CHANGELOG.md` · `RELEASING.md` | 버전 이력과 제출 체크리스트 |
| `.github/` | CI 워크플로우와 이슈·PR 템플릿 |

나머지 경로는 주최측 스타터 키트를 그대로 둔 것입니다.

`analysis/`는 네 파일입니다. `train_linear.py`가 최종 학습기이고, `os2_features.py`와 `os2_policy.py`는 학습 측이 런타임과 같은 특징·배분을 쓰도록 맞춘 이식본입니다. 두 구현이 비트 단위로 같은지는 `tests/test_feature_parity.py`와 `tests/test_learned_guard.py`가 공개 train 전량으로 검사합니다. `cv_gate.py`는 템플릿 그룹 5-fold × 3 seed로 fold 밖 예측을 다시 만들어 가드·margin 후보를 사전 등록 게이트로 판정하는 스크립트로, 입력 split의 그룹 라벨만 읽고 Dev의 결과는 읽지 않습니다.

## 재현과 검증

아티팩트 재학습에는 numpy만 필요합니다. 스크립트가 종료 직전에 런타임 구현으로 같은 문항을 다시 예측해 최대 오차가 1e-9를 넘으면 실패로 중단하며, 현재 아티팩트에서 이 값은 1.67e-15입니다.

```console
PYTHONPATH=src python3 analysis/train_linear.py
```

가드·캘리브레이션 후보를 다시 판정하려면 게이트를 돌립니다. 15 fold 재적합과 후보 격자 평가에 약 1분이 걸립니다.

```console
PYTHONPATH=src python3 analysis/cv_gate.py --guards G0-current --report build/cv-gate-report.json
```

공식 평가 플랫폼은 `linux/arm64`이며, 자원 한도는 CPU 2코어 · 메모리 2 GiB(스왑 없음) · 등급당 90초 · 프로세스 32 · 압축 레이어 합계 1 GiB입니다.

```console
docker build --pull --platform linux/arm64 \
  --file container/Dockerfile --tag ossp-router:local .

PYTHONPATH=src python3 tools/check_runtime.py \
  --image ossp-router:local \
  --report build/runtime-check-report.json
```

테스트는 의존성 없이 돕니다.

```console
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

결정성은 세 층위로 검사합니다. 반복 실행 시 출력 바이트가 일치하고, 문항 순서를 섞거나 episode_id를 바꿔도 배정이 변하지 않으며, 배분의 동률 처리를 문항 위치에서 예측값으로 바꾼 뒤 캘리브레이션 시점 배분과의 동치성을 전수 비교했습니다. 이미지에 무엇이 실리는지는 `.dockerignore` 화이트리스트가 정하고, `tests/test_image_packaging.py`가 진입점에서 도달 가능한 모듈을 정적으로 추적해 누락을 잡습니다.

개발 규율도 적어둘 만합니다. 후보 선택은 train 전용 교차검증으로만 했고, 공개 Dev는 확정된 구성의 예산을 확인하는 용도로만 열었습니다 — 버전을 확정한 뒤 각 1회씩입니다. 캠페인 전체에서 Dev 점수가 후보 선택에 개입한 횟수는 0회입니다. 이 규율을 세운 계기는 초기 버전이 Dev를 반복해서 들여다본 탓에 점수의 1.1pp가 허상이었다는 자체 감사 결과였습니다.

## 참가·제출 절차

주최측 스타터의 절차를 그대로 따릅니다.

1. 저장소를 참가 팀 계정으로 fork합니다.
2. 공개 Train/Dev 자료와 규칙을 확인하고 구현합니다.
3. `self-check`와 컨테이너 실행으로 세 등급의 선택 결과를 확인합니다.
4. 제출할 커밋을 공개하고, 그 커밋에서 `linux/arm64` 이미지를 빌드해 공개 레지스트리에 push합니다.
5. 저장소 루트에 `submission-ossp-skt.json`을 추가해 별도 커밋하고, 그 커밋의 고정 스냅샷 URL을 결과보고서에 기재합니다.

제출 시점부터 평가가 끝날 때까지 fork와 커밋을 별도 권한 없이 열 수 있어야 하며, 수상팀은 수상일로부터 5년 동안 저장소를 공개 상태로 유지해야 합니다. 기술 제출 파일은 `python3 tools/validate_technical_submission.py`로 검증합니다.

## 사용한 오픈소스

제출 이미지 안의 런타임은 파이썬 표준 라이브러리만 씁니다. 절약이 아니라 규칙 대응입니다 — 평가 컨테이너는 네트워크 없이 돌고 압축 이미지 1 GiB 한도를 받으므로, 런타임 의존성을 지우는 쪽이 재현성과 감사 가능성에서 유리했습니다. 이미지 밖의 개발 전 과정은 오픈소스 위에서 이뤄졌습니다.

| 부문 | 도구 |
|---|---|
| 학습 | numpy — dual ridge 폐형해와 특징 행렬 연산 |
| 데이터 준비 | 주최측 실체화 도구와 그 의존성 (`data/sources/requirements-materialize-public-data.txt`) |
| 테스트 | unittest — 의존성 없이 라우터 계약·패키징·저장소 정책까지 검사 |
| 정적 검사 | ruff |
| 라이선스 체계 | REUSE / SPDX — 파일 단위 라이선스 식별과 `reuse lint` 검증 |
| 컨테이너 | Docker BuildKit — `linux/arm64` 재현 빌드, 기반 이미지는 고정 digest의 python alpine |
| CI | GitHub Actions — 테스트·정적 검사·라이선스 검사를 push마다 실행 |

## 문서

과제 규격(규칙 · 채점 · 런타임 · 데이터 · 제출 · 집행)은 `docs/`의 대문자 파일명 주최측 문서를 그대로 따릅니다. 이 저장소가 직접 작성한 문서는 다음과 같습니다.

| 문서 | 내용 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 계층 구조·아티팩트 형식·결정성 보장 |
| [docs/decisions.md](docs/decisions.md) | 판정 절차와 채택·기각 결정 기록 |
| [CHANGELOG.md](CHANGELOG.md) | 버전 이력 |
| [RELEASING.md](RELEASING.md) | 제출·릴리스 체크리스트 |

실험 전문과 게이트 원장은 별도 문서 저장소에 있습니다.

## 라이선스

프로젝트가 직접 작성한 코드와 문서는 [Apache License 2.0](LICENSE)으로 제공합니다. 이 라이선스는 제3자 벤치마크 자료를 재라이선스하지 않으며, 자료별 조건은 [DATA_LICENSES.md](DATA_LICENSES.md)에 따로 기록합니다. 기여 정책은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하십시오.

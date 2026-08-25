<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# Efficient LLM Routing Challenge — 예산 제약 라우터 R1.4

[![tests](https://github.com/yuJunhyk/ossp-2026-llm-router-challenge/actions/workflows/test.yml/badge.svg)](https://github.com/yuJunhyk/ossp-2026-llm-router-challenge/actions/workflows/test.yml)

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
| 4 | 소수·인수분해 어휘와 7자리 이상 정수가 함께 나오면 추론 모델의 예측 점수를 중간 모델 수준으로 눌러 승급 유인을 차단 | train에는 비용이 최저가의 천 배에서 만 배에 이르는 컨텍스트 소진 문항이 26건 있고 프롬프트만으로는 예측되지 않음 — 어휘 규칙이 닿는 범위는 원천에서 막음 |
| 5 | 예측 비용에 비관 쐐기 `exp(β·σ)`를 곱함 — β는 세 등급 모두 1.0 | β를 0.5로 낮춘 후보들이 새로 뽑은 fold 검사에서 다섯 번 예산을 터뜨림. 반쪽 쐐기는 고비용 문항 두어 건이 한 배치에 몰리는 사고를 견디지 못함 |
| 6 | 예산 상한에 margin을 곱함 — 0.94 · 1.08 · 0.96, Premium은 문항 800개 이상 배치에서 1.08 | β=1.0이 계획 비용을 두 배 가까이 부풀리므로 Balanced는 지갑을 그만큼 되돌려 열고, 예산이 문항 수에 비례하는 규칙 구조에 맞춰 큰 배치에서만 Premium 지갑을 넓힘. 배치가 작아지면 자동으로 보수 쪽으로 접힘 |
| 7 | 승급 증분을 점수 이득 ÷ 비용 순으로 사들이는 볼록껍질 그리디 배낭 | 동률은 문항 위치가 아니라 예측값으로 끊어, 문항을 섞거나 ID를 바꿔도 배정이 변하지 않음 |

라우터는 문항 ID·입력 순서·split·과제명을 읽지 않습니다. 모델 선택 함수가 받는 것은 프롬프트 본문과 실행 등급뿐입니다. 싼 모델을 먼저 돌려보고 결과를 본 뒤 승급하는 방식은 규칙이 금지하며, 이 라우터에는 그런 경로가 없습니다.

## 왜 이렇게 정했나

**신호가 없으면 배우지 말고 버리고, 신호가 실재하면 출처를 가리지 않습니다.** 3단계의 상수 교체가 첫 번째 채택 개선입니다. 비싼 모델로 바꿨을 때 정답이 뒤집히는 정도를 문항별로 예측하려던 시도가 참값과 상관 0.033에 그쳤고, 그 노이즈를 배분에 먹이는 것보다 학습 평균 상수 하나로 갈음하는 편이 나았습니다. 두 번째 채택 개선은 반대 방향입니다 — 팀 동료가 설계한 서술형 밀집 특징 27종은 같은 사전 등록 게이트에서 판별 개선이 재현되어 그대로 들어왔습니다.

**꼬리는 예측이 아니라 정책으로 막습니다.** 비용의 두꺼운 꼬리를 더 정밀한 예측으로 잡으려는 시도 — 분위수 상한, 유형별 분산 분리 — 는 실측에서 전부 기각됐습니다. 조건부 비용 신호 자체는 실재했지만(잔차 상관 +0.43) 같은 margin에서 예산을 열 번 터뜨렸습니다. 쐐기의 강도도 같은 방식으로 정해졌습니다. β를 0.5로 낮춘 후보들은 캘리브레이션 검사를 통과하고도 새로 뽑은 fold에서 다섯 번 예산을 터뜨렸는데, 원인은 다이얼이 아니라 절차였습니다 — 관측된 최악치에 맞춰 margin을 고르면 당선자는 언제나 관측 경계에 앉고, 유한 표본의 최악은 진짜 최악을 과소평가합니다. 그래서 판정 기준에 선택 밖 fold와 부트스트랩을 추가하고 β는 세 등급 모두 1.0으로 올렸습니다.

**여유는 취향이 아니라 필요조건입니다.** 주최측 baseline 하나가 공개 Dev에서 예산의 99.6%를 쓰고도 채점용 평가셋에서 한도를 넘겨 0점 처리된 전례가 있습니다. 관측된 비용 이동은 약 +5.4%였습니다. 그래서 이 라우터의 Dev 예산 판정은 사용률에 1.054를 곱한 값까지 한도 안이어야 통과로 정의했고, 세 등급 모두 통과했습니다. 실제 여유는 그보다 큽니다 — 가장 빠듯한 Fast가 +12%, Premium은 +24%의 이동을 견딥니다. Premium이 예산의 80.3%만 쓰고 남기는 몫은 계산 착오가 아니라 파산을 막는 보험료입니다.

## 한계

정직하게 적어둡니다.

- **Fast 등급 회수율이 15.4%로 가장 낮은데 가중치는 0.4로 가장 큽니다.** 예산이 1.25배뿐이라 살 수 있는 승급의 절대량이 적은 구조적 제약이며, 라우터 품질로 메울 수 있는 종류가 아닙니다.
- **Premium 실지출이 한도의 80.3%에 그칩니다.** β=1.0이 계획 비용을 실제 기대의 두 배 가까이로 부풀리기 때문입니다. 큰 배치에서 지갑을 넓히는 margin 분기로 이전 구성(60.7%)보다 격차를 좁혔지만, 나머지는 위에 적은 보험료입니다.
- **예측기 선택의 근거가 근소합니다.** 재대결에서 1위와 3위의 차이가 0.0013으로 교차검증 노이즈 안에 있습니다. 사전 등록한 규칙과 레이턴시·단순성 근거를 기록으로 남기는 것 외에 더 할 수 있는 게 없었습니다.
- **공개 라벨이 1,760문항뿐입니다.** 사전 등록 게이트로 측정한 개선 후보 스무 경로 남짓 가운데 채택은 넷 — 상수 대체, 밀집 특징, 전 등급 재캘리브레이션, margin 규모 분기 — 뿐이고, 기각된 여럿의 사인은 아이디어의 결함이 아니라 라벨 부족이었습니다. 소형 인코더 임베딩은 신호가 실재함을 확인했지만(분산 -1.84%) 배분 점수로 환산되지 않았습니다.

## 저장소 구조

| 경로 | 내용 | 출처 |
|---|---|---|
| `src/ossp_router/learned_*.py` | 제출 이미지에 실리는 라우터 런타임. 표준 라이브러리만 사용 | 우리 |
| `src/ossp_router/resources/learned-router.v1.json` | 학습 결과 아티팩트 1.11 MB | 우리 |
| `analysis/` | 학습 파이프라인 세 파일. numpy만 필요하며 이미지에는 들어가지 않음 | 우리 |
| `tests/test_learned_*.py` · `test_feature_parity.py` · `test_image_packaging.py` · `test_runtime_edges.py` | 라우터 계약·특징 동등성·패키징 테스트 | 우리 |
| `docs/architecture.md` · `decisions.md` · `roadmap.md` | 설계·결정 기록·로드맵 (docs/ 안에서 소문자 파일명이 우리 문서) | 우리 |
| `CHANGELOG.md` · `RELEASING.md` | 버전 이력과 제출 체크리스트 | 우리 |
| `.github/` | CI 워크플로우와 이슈·PR 템플릿 | 우리 |
| `src/ossp_router/` 나머지 | 프로토콜·채점·CLI·실행 진입점 | 주최측 |
| `baselines/` · `container/` · `configs/` · `data/` · `docs/`(대문자 문서) · `schemas/` · `tools/` | 스타터 키트 | 주최측 |

`analysis/`는 세 파일입니다. `train_linear.py`가 최종 학습기이고, `os2_features.py`와 `os2_policy.py`는 학습 측이 런타임과 같은 특징·배분을 쓰도록 맞춘 이식본입니다. 두 구현이 비트 단위로 같은지는 `tests/test_feature_parity.py`가 공개 train 전량으로 검사합니다.

## 아티팩트 재학습

numpy만 있으면 됩니다.

```console
PYTHONPATH=src python3 analysis/train_linear.py
```

기본값으로 공개 train 1,760문항을 읽어 `src/ossp_router/resources/learned-router.v1.json`을 다시 씁니다. 학습이 끝나면 스크립트가 런타임 구현으로 같은 문항을 다시 예측해 값이 일치하는지 대조하며, 최대 오차가 1e-9를 넘으면 실패로 중단합니다. 현재 아티팩트에서 이 값은 1.67e-15입니다.

## 컨테이너

공식 평가 플랫폼은 `linux/arm64`입니다.

```console
docker build --pull --platform linux/arm64 \
  --file container/Dockerfile --tag ossp-router:local .

PYTHONPATH=src python3 tools/check_runtime.py \
  --image ossp-router:local \
  --report build/runtime-check-report.json
```

| 자원 | 한도 |
|---|---|
| CPU | 2 코어 |
| 메모리 | 2 GiB, 스왑 없음 |
| 등급당 실행 시간 | 90초 |
| 프로세스·스레드 | 32 |
| OCI 압축 레이어 합계 | 1 GiB |

이미지에 무엇이 들어가는지는 `.dockerignore`가 전부 거부한 뒤 필요한 파일만 다시 허용하는 방식으로 정합니다. 이 목록에서 파일 하나가 빠져 세 등급이 모두 0점이 될 뻔한 적이 있어서, `tests/test_image_packaging.py`가 진입점에서 도달 가능한 모듈을 정적으로 추적해 누락을 잡습니다.

## 검증

```console
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

결정성은 세 층위로 검사합니다. 반복 실행 시 출력 바이트가 일치하고, 문항 순서를 섞거나 episode_id를 바꿔도 배정이 변하지 않으며, 배분의 동률 처리를 문항 위치에서 예측값으로 바꾼 뒤 캘리브레이션 시점 배분과의 동치성을 전수 비교했습니다.

개발 규율도 적어둘 만합니다. 후보 선택은 train 전용 교차검증으로만 했고, 공개 Dev는 확정된 구성의 예산을 확인하는 용도로만 열었습니다 — 버전을 확정한 뒤 각 1회씩입니다. 캠페인 전체에서 Dev 점수가 후보 선택에 개입한 횟수는 0회입니다. 이 규율을 세운 계기는 초기 버전이 Dev를 반복해서 들여다본 탓에 점수의 1.1pp가 허상이었다는 자체 감사 결과였습니다.

## Quickstart: baseline에서 시작하기

주최측이 제공하는 baseline 네 종 — 전량 최저가, 프롬프트 휴리스틱, 특징 예산, 해시 정규식 — 의 실행법과 설명은 [baselines/README.md](baselines/README.md)에 있습니다. 이 저장소의 `router-run` 진입점은 위 라우터로 배선돼 있으므로, baseline을 돌려보려면 각 baseline 스크립트를 직접 실행하면 됩니다.

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

과제 규격은 주최측 문서를 그대로 따릅니다.

| 문서 | 내용 |
|---|---|
| [docs/CHALLENGE_RULES.md](docs/CHALLENGE_RULES.md) | 과제 규칙 |
| [docs/SCORING.md](docs/SCORING.md) | 점수 계산 |
| [docs/RUNTIME.md](docs/RUNTIME.md) | 컨테이너 실행 규격 |
| [docs/DATA_CARD.md](docs/DATA_CARD.md) | 데이터 카드 |
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | 제출 안내 |
| [docs/ENFORCEMENT.md](docs/ENFORCEMENT.md) | 규칙 위반 처리 |

이 저장소가 직접 작성한 문서는 다음과 같습니다.

| 문서 | 내용 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 계층 구조·데이터 흐름·결정성 보장 |
| [docs/decisions.md](docs/decisions.md) | 사전 등록 게이트 프로토콜과 채택·기각 결정 기록 |
| [docs/roadmap.md](docs/roadmap.md) | 이후 발전 방향 |
| [CHANGELOG.md](CHANGELOG.md) | 버전 이력 |
| [RELEASING.md](RELEASING.md) | 제출·릴리스 체크리스트 |

실험 전문과 게이트 원장은 별도 문서 저장소에 있습니다. 결과보고서는 설계·검증·결과 분석·한계를, 버전 회고는 채택하지 않은 경로들의 사인을 각각 다룹니다.

## 라이선스

프로젝트가 직접 작성한 코드와 문서는 [Apache License 2.0](LICENSE)으로 제공합니다. 이 라이선스는 제3자 벤치마크 자료를 재라이선스하지 않으며, 자료별 조건은 [DATA_LICENSES.md](DATA_LICENSES.md)에 따로 기록합니다. 기여 정책은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하십시오.

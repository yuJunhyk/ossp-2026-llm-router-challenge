<!--
SPDX-FileCopyrightText: Copyright 2026 yuJunhyk
SPDX-License-Identifier: Apache-2.0
-->

# RELEASING — 제출 체크리스트

이 저장소는 대회 제출물이라 릴리스가 곧 제출입니다. 아래 목록은 두 실측 사고에서 나왔습니다. 하나, 이미지 화이트리스트(`.dockerignore`)에서 런타임 파일 하나가 빠져 세 등급 전부가 0점이 될 뻔했습니다. 둘, 주최측 baseline 하나는 공개 Dev 예산의 99.6%를 쓰고도 채점셋에서 한도를 넘겨 0점이 됐습니다. 제출 직전 확인은 기억이 아니라 목록으로 합니다.

## 1. 버전·수치 문자열 등장 위치 (전수)

구성이 바뀌면 아래를 모두 갱신합니다.

| 파일 | 위치 |
|---|---|
| `README.md` | 제목(H1), 결과 표, 회수율, 등급당 실행 시간, 동작 표의 특징 수·β·margin, 한계 절 |
| `CHANGELOG.md` | 신규 버전 절 |
| `docs/decisions.md` | 최종 구성 절 |
| `src/ossp_router/resources/learned-router.v1.json` | `tier_config` — 재학습으로만 생성하고 손으로 고치지 않습니다 |
| `submission-ossp-skt.json` | `commit_sha` · `image_digest` — 제출 커밋마다 |

## 2. 제출 전 검증 (순서대로)

1. `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'` 전체 통과
2. 아티팩트를 재생성했다면 `PYTHONPATH=src python3 analysis/train_linear.py` — 런타임 대조 오차가 1e-9를 넘으면 스스로 중단됩니다
3. `reuse lint` 통과
4. `docker build --pull --platform linux/arm64 --file container/Dockerfile --tag ossp-router:local .`
5. `PYTHONPATH=src python3 tools/check_runtime.py --image ossp-router:local --report build/runtime-check-report.json` 세 등급 PASS
6. 컨테이너와 호스트의 공개 Dev 결정 전량 일치 확인

## 3. 제출 절차 (순서 제약)

이미지는 제출 JSON이 없는 커밋에서 빌드해야 합니다. 순서를 바꾸면 `commit_sha`와 digest가 어긋납니다.

1. 최종 코드 커밋을 확정·공개하고, 그 커밋에서 `linux/arm64` 이미지를 빌드해 공개 레지스트리에 push한 뒤 전체 digest를 확인합니다.
2. `submission-ossp-skt.json`을 작성합니다. `repository_url`은 커밋 경로 없는 저장소 기본 URL, `commit_sha`는 1의 빌드 커밋 40자리 소문자, `image_digest`는 `레지스트리/저장소@sha256:<64자리>` 전체 형식입니다.
3. `python3 tools/validate_technical_submission.py`가 exit 0인지 확인하고, JSON만 담은 별도 커밋을 push합니다.
4. 3의 커밋 고정 스냅샷 URL(`.../tree/<전체 SHA>`)을 결과보고서의 프로젝트 등록 URL에 기재합니다.
5. 결과보고서 원본(한글/Word)과 PDF를 대회 사이트에 업로드합니다. 마감은 2026-08-27 18:00 KST입니다.

## 4. 동결 원칙

공개 Dev 예산 판정을 통과한 구성은 그 시점부터 동결입니다. Dev 수치를 본 뒤의 재튜닝은 이 저장소가 자체 감사로 실측한 오염(점수 1.1pp 허상)의 재발입니다.

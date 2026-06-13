# 말해주차 (Smart Parking)

YOLO 영상분석으로 주차 점유를 자동 감지하고, 그 현황을 **웹과 안드로이드 앱**에서 지도·검색·음성으로 확인하는 스마트 주차 시스템입니다.

## 구성

| 폴더 | 설명 | 기술 |
|---|---|---|
| `fastapi/` | YOLO 추론 서버 — 영상에서 차량 탐지 → 슬롯 점유 계산 (`/status`) | Python, FastAPI, Ultralytics YOLO, OpenCV |
| `springboot/` | 캠퍼스/건물/주차장 API + 웹 UI + 인증 + 음성·검색 프록시 | Java 17, Spring Boot 4, JPA, H2, Spring Security |
| `P_Application/` | 안드로이드 앱 (위 백엔드를 호출하는 모바일 클라이언트) | Android(Java), Retrofit, 네이버 지도 SDK |
| `docs/` | 설계·구현 계획 문서 | — |

> 셋은 코드 공유가 아니라 **HTTP REST API**로 연결됩니다.

## 주요 기능

- **점유 감지**: YOLO가 영상에서 차량을 탐지해 주차장별 빈자리/총칸 계산
- **지도 기반 등록(웹)**: 지도에서 위치를 클릭해 건물·주차장을 동적 등록(영상 업로드), 삭제. 데이터는 영속 저장
- **장소 검색**: 장소명 검색(네이버 지역검색)으로 지도 이동
- **음성 질의**: "AI공학관 빈자리 있어?" 음성 질문 → LLM(Gemini)이 현황 기반 자연어 답변 → 음성 출력 (웹·앱)
- **앱**: 지도 핀→주차장별 현황(여유/혼잡/만차), 슬롯 사진 오버레이, 즐겨찾기, 내 주차위치 저장/추적, 빈자리 알림·알림함

## 실행 방법

### 백엔드 (Spring Boot + FastAPI)

루트의 통합 스크립트로 한 번에 실행합니다.

```bash
./run.sh
```

- Spring Boot 웹: `http://localhost:8080/`
- FastAPI(YOLO): `http://localhost:8000/`
- `run.sh`가 Java 17 경로와 네이버/Gemini API 키(환경변수)를 자동으로 로드합니다.

개별 실행:

```bash
# Spring Boot
cd springboot && ./gradlew bootRun
# FastAPI (별도 터미널, 영상 분석 창까지: SHOW_GUI=1)
cd fastapi/video_test && ./venv/bin/python -m uvicorn server0:app --host 0.0.0.0 --port 8000
```

### 안드로이드 앱 (P_Application)

Android Studio에서 `P_Application/`을 열고 에뮬레이터/실기기로 실행합니다.

- 앱은 에뮬레이터 기준 `http://10.0.2.2:8080`(= PC의 localhost)으로 백엔드를 호출합니다.
- 따라서 앱 실행 전 **백엔드(`./run.sh`)가 켜져 있어야** 합니다.

## 환경변수 (API 키)

키는 소스에 포함하지 않고 환경변수로 관리합니다. `.env.example` 참고.

| 변수 | 용도 |
|---|---|
| `SMARTPARKING_NAVER_MAP_CLIENT_ID` | 네이버 지도 표시 |
| `SMARTPARKING_NAVER_SEARCH_CLIENT_ID` / `_SECRET` | 장소명 검색(지역검색 API) |
| `SMARTPARKING_GEMINI_API_KEY` | 음성 질의 답변 생성(Gemini) |
| `SMARTPARKING_JWT_SECRET` | 로그인 JWT 서명 |

## 주요 API (요약)

- 조회: `GET /api/campus/map`, `GET /api/campus/buildings/{id}`
- 등록/삭제: `POST /api/buildings`, `POST /api/buildings/{id}/parking-lots`, `DELETE ...`
- 검색: `GET /api/geo/search?query=`
- 음성: `POST /api/voice/ask`
- 인증: `POST /auth/login`, `POST /auth/register`
- 내 정보(로그인 필요): `/api/me/parking-location`, `/api/me/alert-rules`, `/api/me/notifications`

자세한 설계는 `docs/` 참고.

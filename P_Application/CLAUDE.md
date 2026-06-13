# P_Application — 안드로이드 앱 (스마트 주차 / "말해주차")

> 이 문서는 **백엔드 쪽에서 작업하던 Claude가 2026-06-13에 작성·검증**했습니다.
> 목적: Android Studio에서 이 앱을 작업하는 Claude가 **백엔드 REST API 계약**을 바로 알 수 있도록.
> 아래 모든 엔드포인트/필드는 **실제 백엔드 DTO와 대조 완료**(2026-06-13). 단, 백엔드가 계속 바뀌면 맨 아래 "검증"으로 재확인하세요.

## 이 프로젝트가 뭔지

- **네이티브 안드로이드 앱 (Java)**. 스마트 주차 시스템의 **경량 모바일 클라이언트**.
- 기능: ① 지도에서 주차장 **점유 현황** 보기 → ② **음성으로 주차 현황 질의**(STT→백엔드→TTS) → ③ **내 주차위치 저장** → ④ **빈자리 알림**.
- 앱은 **자체 서버가 없음**. 옆의 **Spring Boot REST API**를 호출하는 클라이언트일 뿐.
- 앱은 **조회·음성·내위치·알림만** 사용하는 클라이언트다. 건물/주차장 **등록·삭제(POST /api/buildings 등)는 웹/관리용**이라 앱에서 쓰지 않는다.

## 레포/브랜치 구조 (헷갈리기 쉬움)

- 이 폴더(`/Users/leehnsong/P-app/P_Application`)는 **P-Project 레포의 git worktree**이며 브랜치는 `feature/map_api-app`.
- **백엔드 코드(Spring Boot/FastAPI)는 다른 worktree** `~/P-Project` (브랜치 `feat/map-pin-registration`)에 있음 — 이 앱 폴더에선 안 보임.
- 즉 앱과 백엔드는 **코드 공유가 아니라 HTTP API로만** 연결됨. POJO는 아래 JSON 모양에 맞춰 만들면 됨.
- ⚠️ **백엔드 실행 전제**: 점유 현황과 **음성 API는 `feat/map-pin-registration` 브랜치에 구현돼 있다.** 앱이 정상 동작하려면 그 브랜치의 백엔드가 떠 있어야 한다(또는 main에 머지 후 실행). 음성은 추가로 백엔드에 `SMARTPARKING_GEMINI_API_KEY` 환경변수가 설정돼 있어야 답변이 온다.

## 백엔드 통신 기본

- 프로토콜: **HTTP + JSON REST**, 인증은 **JWT** (`Authorization: Bearer <token>`).
- Base URL (개발):
  - **PC 터미널에서 테스트(curl 등)**: `http://localhost:8080/`
  - **안드로이드 에뮬레이터(앱/adb)**: `http://10.0.2.2:8080/`  (= 에뮬레이터에서 PC의 localhost를 가리키는 특수 주소)
  - **실기기**: PC의 LAN IP (예: `http://192.168.0.x:8080/`)
  - ※ `10.0.2.2`는 **에뮬레이터 안에서만** 의미 있음. PC 셸에서 `10.0.2.2`로 curl하면 실패하니 PC에선 `localhost`를 써라.
- 권장 스택: **Retrofit + OkHttp + Gson**. STT/TTS는 **안드로이드 네이티브**(`SpeechRecognizer` / `TextToSpeech`, `ko-KR`).
- 매니페스트 권한: `INTERNET`, (음성용) `RECORD_AUDIO`.
- HTTP(평문) 호출이므로 디버그 빌드에서 `usesCleartextTraffic=true`(또는 `10.0.2.2`/LAN IP 도메인 허용 network-security-config) 필요.
- CORS는 앱과 무관(브라우저 전용). 네이티브 호출엔 영향 없음.

## Gson 매핑 규칙 (중요)

- **POJO에 적은 필드명만 JSON과 일치하면 됨.** 응답에는 문서에 안 적은 필드가 더 있을 수 있는데 **Gson이 자동 무시**한다. 그러니 필요한 필드만 POJO에 두면 된다.
- 키 형식(`bldg-...`)에 의존하지 말고 **항상 `id`로 식별**하라. 건물/주차장은 동적 생성되어 `mapKey`는 `bldg-a3636389`, `partitionKey`는 `bldg-a3636389_1` 같은 자동 생성 값이다(아래 예시의 `gachon_ai`는 옛 예시일 뿐 실제 값 아님).

## 앱이 쓰는 엔드포인트

### 1) 캠퍼스 지도 + 점유 (지도 화면 핵심)
`GET /api/campus/map`  — 인증 불필요
```jsonc
{
  "campus": { "id": 1, "name": "...", "centerLat": 37.45, "centerLng": 127.13, "defaultZoom": 17 },
  "buildings": [
    {
      "id": 1, "name": "AI공학관", "mapKey": "bldg-a3636389",
      "lat": 37.45, "lng": 127.13, "sortOrder": 1,
      "parkingLots": [
        {
          "id": 1, "name": "지하 1층", "partitionKey": "bldg-a3636389_1",
          "summary": {
            "status": "AVAILABLE",     // "AVAILABLE" | "FULL" | "NO_DATA"
            "totalSlots": 41,          // 총 칸
            "availableSlots": 4,       // 빈자리
            "disabledAvailable": 0,    // 장애인석 빈자리
            "lastUpdate": 1718000000.0 // epoch '초'(실수, nullable)
          }
        }
      ]
    }
  ]
}
```
→ 지도 마커는 `buildings[].lat/lng`, 점유 표시는 `parkingLots[].summary`. (이 응답의 parkingLots엔 slots가 비어 있음 — 슬롯이 필요하면 2)번 호출)

### 2) 건물 상세 (마커/주차장 탭 시)
`GET /api/campus/buildings/{buildingId}`  — 인증 불필요
```jsonc
{
  "campus":   { ... },                                  // 1)과 동일 구조
  "building": { "id": 1, "name": "AI공학관", "mapKey": "bldg-a3636389",
                "lat": 37.45, "lng": 127.13, "sortOrder": 1, "parkingLots": [ ... ] },
  "parkingLots": [
    {
      "id": 1, "name": "지하 1층", "partitionKey": "bldg-a3636389_1",
      "summary": { ... },                               // 1)과 동일
      "slots": [                                         // 상세에는 슬롯별 정보 포함
        { "partitionKey": "bldg-a3636389_1", "slotId": 3, "type": "normal",
          "status": "available", "center": [283.0, 154.0] }   // status: "available" | "occupied"
      ]
    }
  ]
}
```
→ 슬롯 목록은 **최상위 `parkingLots[].slots`** 를 사용. `slots[].center`는 `[x, y]` 픽셀 좌표(double).

### 3) 음성 주차 질의 ✅ 구현·검증 완료
`POST /api/voice/ask`  — 인증 불필요(공개)
```jsonc
// 요청
{ "question": "AI공학관 빈자리 있어?" }
// 응답
{ "answer": "현재 지하 1층은 4자리 비어 있어요." }
```
- 앱 흐름: 네이티브 STT로 음성→텍스트 → 이 API에 `question` 전송 → `answer` 받기 → 네이티브 TTS로 음성 출력.
- 빈 `question`은 **400**. 백엔드 LLM(Gemini) 호출 실패 시에도 200 + 안전한 폴백 문장이 온다.
- **전제**: 백엔드가 `feat/map-pin-registration` 브랜치(+ `SMARTPARKING_GEMINI_API_KEY` 설정)로 떠 있어야 한다.

### 4) 인증
- `POST /auth/login`  요청 `{ "username", "password" }` → 응답 `{ "token", "username" }`
- `POST /auth/register`  요청 `{ "username", "password" }` → 응답: 문자열(예: `REGISTER_SUCCESS` / `USER_EXISTS`)
- 이후 사용자 전용 API(`/api/me/**`)는 `Authorization: Bearer <token>` 헤더 필수.
- 점유 조회/음성 질의는 **로그인 없이** 가능. 단 **내 주차위치·빈자리 알림(아래 5~7)** 은 **로그인 필수**.

### 5) 내 주차위치 저장  🔒 로그인 필요
- `GET /api/me/parking-location/current` → 현재 저장된 위치 (없으면 **204 No Content**)
- `POST /api/me/parking-location`  요청 `{ "parkingLotId", "slotId", "vehicleLabel", "memo" }`
  - 이전 활성 위치는 자동 비활성화되고 새 위치가 활성됨
- `DELETE /api/me/parking-location/current` → 주차 종료(해제, 비활성 처리)

현재 위치 응답 예 (실제 필드명 확인됨):
```jsonc
{ "id": 1, "parkingLotId": 1, "parkingLotName": "지하 1층", "partitionKey": "bldg-a3636389_1",
  "slotId": 3, "vehicleLabel": "내 차", "memo": "기둥 옆",
  "active": true, "savedAt": "2026-06-13T10:00:00", "releasedAt": null }
```
→ `savedAt`/`releasedAt`은 ISO-8601 LocalDateTime 문자열(타임존 없음). 없으면 `null`.

### 6) 빈자리 알림 규칙  🔒 로그인 필요
- `GET /api/me/alert-rules` → 내 규칙 목록
- `POST /api/me/alert-rules`  요청 `{ "parkingLotId", "minimumAvailableSlots", "enabled" }`
- `PUT /api/me/alert-rules/{ruleId}` (수정, 보통 enabled 토글),  `DELETE /api/me/alert-rules/{ruleId}` (삭제, 204)

규칙 응답 예 (실제 필드명 확인됨):
```jsonc
{ "id": 1, "parkingLotId": 1, "parkingLotName": "지하 1층",
  "minimumAvailableSlots": 5, "enabled": true,
  "lastKnownAvailableSlots": 3, "lastTriggeredAt": null }
```
동작: 서버 스케줄러가 점유를 감시하다 **"빈자리 ≥ 임계값"으로 전환되는 순간** 인앱 알림을 생성.

### 7) 알림 조회  🔒 로그인 필요
- `GET /api/me/notifications` → `[{ "id", "title", "message", "category", "read", "createdAt", "readAt" }]`
  - `read`는 boolean(읽음 여부). `category`/`readAt` 추가 필드 있음.
- `GET /api/me/notifications/unread-count` → `{ "unreadCount": 0 }`  (long)
- `PATCH /api/me/notifications/{notificationId}/read` → 읽음 처리

⚠️ **현재는 "인앱 알림(폴링)" 방식**: 앱이 위 API를 주기적으로 호출해 화면에서 확인.
OS 푸시(앱이 꺼져 있어도 뜨는 알림)는 **미구현** → 그건 **FCM(Firebase Cloud Messaging) 추가가 필요**(범위 밖). 데모는 인앱 폴링으로 충분.

### (참고) 기타
`GET /api/parking/status`(원시 점유 캐시, 없으면 204), `GET /api/ui/config`(네이버 지도 클라이언트ID 등). 필요할 때만.

## 작업 규칙

- 응답 JSON ↔ Retrofit POJO는 **필드명만 일치**시키면 됨(나머지는 Gson이 무시). 식별은 항상 `id`로.
- 네이버 지도는 백엔드 웹에서만 쓰던 것 — 앱에선 **안드로이드용 네이버 지도 SDK**를 별도로 붙여야 한다(웹 키와 별개로 안드로이드 환경/클라이언트ID 설정 필요할 수 있음). 음성/조회 API는 지도와 무관하게 먼저 붙일 수 있음.
- 커밋은 `feature/map_api-app` 브랜치로. push 전 `git pull origin feature/map_api-app` 먼저(필요 시 백엔드 머지 상태도 확인).

## ✅ 검증 (의존하기 전에)

이 문서는 2026-06-13 백엔드 DTO와 대조했지만, 백엔드가 또 바뀌면 어긋날 수 있다. 확인법:

1. **백엔드를 먼저 실행**해야 한다. 백엔드 폴더(`~/P-Project`)에서:
   ```bash
   cd ~/P-Project && ./run.sh        # 또는 cd ~/P-Project/springboot && ./gradlew bootRun
   ```
   (`run.sh`가 Java/네이버/Gemini 키를 ~/.zshrc에서 자동 로드)
2. **PC 터미널에서** 응답 모양 보기(에뮬레이터가 아니라 PC이므로 `localhost`):
   ```bash
   curl -s http://localhost:8080/api/campus/map | python3 -m json.tool
   curl -s -X POST http://localhost:8080/api/voice/ask \
     -H "Content-Type: application/json" -d '{"question":"빈자리 있어?"}'
   ```
3. 또는 백엔드 DTO 직접 확인: `~/P-Project/springboot/src/main/java/com/smartparking/server/dto/`
   (`CampusMapResponse`, `BuildingDetailResponse`, `ParkingLotView`, `VoiceAskRequest/Response`,
    `ParkingLocationResponse`, `ParkingAlertRuleResponse`, `InAppNotificationResponse`, `UnreadCountResponse`)

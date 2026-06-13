# 지도 핀 기반 건물·주차장 동적 등록 설계

- 작성일: 2026-06-13
- 상태: 승인 대기
- 관련 영역: `springboot/` (백엔드 + 웹 UI), `fastapi/video_test/` (영상 스캔, 변경 없음)

## 1. 배경 / 문제

현재 캠퍼스 건물과 좌표는 [CampusDataInitializer](../../../springboot/src/main/java/com/smartparking/server/config/CampusDataInitializer.java)에
**하드코딩된 시드**로 들어간다. 지도 마커는 이 고정 건물 5개에서만 나오고, H2 인메모리 DB라 재시작할 때마다 시드가 다시 생성된다.

사용자는 **지도에서 원하는 위치를 직접 찍어(핀) 건물을 만들고, 그 안에 주차장을 등록**하고 싶다.
즉 위치를 코드가 아니라 사용자가 런타임에 지정하고, 등록한 데이터는 재시작 후에도 유지되어야 한다.

## 2. 목표 / 비목표

### 목표
- 지도 빈 영역 클릭 → 그 좌표에 **건물 동적 생성** (사용자는 이름만 입력)
- 건물에 **주차장 추가 + 영상 업로드** (partitionKey 자동 생성, 기존 YOLO 파이프라인 연결)
- 잘못 찍은 핀 정리를 위한 **건물/주차장 삭제**
- 검색으로 대략 위치 이동 후 핀 찍기 (**NCP Geocoding**)
- **하드코딩 건물 시드 제거** + **영속 저장**(H2 파일 모드)으로 등록 데이터 유지

### 비목표 (이번 범위 아님)
- 슬롯(주차 칸) 정의 자동화 — 기존 맵 빌더 GUI 그대로 재사용 (추후 자동화 여지)
- 관리자 역할(role) 도입 — 로그인 사용자면 누구나 등록/삭제
- FastAPI(YOLO) 로직 변경 — 영상 5초 스캔/30프레임 추론 그대로 활용
- 장소/POI 키워드 검색 — 이번엔 Geocoding(주소·지명)만

## 3. 선택한 접근: DB 우선 (A안)

등록 API가 **건물·주차장 레코드를 DB에 직접 생성**하고, 업로드 영상은 `videos/{partitionKey}_video.mp4`로 저장한다.
기존 파일시스템 자동 스캔([ParkingLotAssetSyncService](../../../springboot/src/main/java/com/smartparking/server/service/ParkingLotAssetSyncService.java))은
"DB에 없는 영상만 보조로 채우는" 역할로 남긴다(하위호환). 하드코딩 건물 시드는 제거하고 캠퍼스(지도 중심) 시드만 유지한다.

- 대안 B(파일시스템 우선): 좌표를 사이드카 파일로 흩뿌려 관리가 지저분 → 기각
- 대안 C(자동 스캔 제거): 가장 단순하나 "영상 떨구면 자동 생성"되던 기존 동작 상실 → 기각

## 4. 데이터 모델 (스키마 변경 없음)

핀=건물이므로 엔티티 스키마는 그대로 두고 **생성 경로만** 바꾼다.

| 엔티티 | 변경 | 비고 |
|---|---|---|
| `Campus` | 시드 유지 | 지도 중심(가천대) 1개만 "없을 때 생성" |
| `Building` | 스키마 무변경 | 이미 `lat/lng/name/mapKey/sortOrder` 보유. 하드코딩 시드 제거 → API로 동적 생성 |
| `ParkingLot` | 스키마 무변경 | 건물 FK·`partitionKey`·`slotLayoutJson` 그대로. API로 동적 생성 |

### 식별자 자동 생성 (사용자 미입력)
- 건물 `mapKey`: 고유 슬러그 자동 생성(예: `bldg-<짧은난수>`). 사용자는 **건물 이름만** 입력.
- 주차장 `partitionKey`: 건물 추가 시 `{mapKey}_{순번}` 형태로 자동 생성. 업로드 영상은 `videos/{partitionKey}_video.mp4`로 저장.
- 둘 다 유일성 보장(중복 시 재생성/순번 증가).

## 5. 백엔드 API

모두 로그인(JWT) 필요. 조회는 비로그인 허용.

| 메서드 · 경로 | 역할 | 입력 | 응답 |
|---|---|---|---|
| `POST /api/buildings` | 핀으로 건물 생성 | `{ name, lat, lng }` | 생성된 building (mapKey 자동) |
| `DELETE /api/buildings/{buildingId}` | 건물 삭제 | — | 204; 하위 주차장 + 영상/슬롯 파일 함께 정리 |
| `POST /api/buildings/{buildingId}/parking-lots` | 주차장 추가 + 영상 업로드 | `multipart`: `name`, `video`(필수), `image`(선택) | 생성된 parkingLot |
| `DELETE /api/parking-lots/{parkingLotId}` | 주차장 삭제 | — | 204; 영상/슬롯/이미지 파일 함께 정리 |

### 기존 재사용 (변경 없음)
- 슬롯 정의: `POST /api/parking-lots/{id}/map/upload`(사진) + `POST .../map/build`(맵 빌더 GUI)
- 현황: FastAPI `/status`가 새 partitionKey 영상을 5초 스캔으로 자동 인식 → Spring Boot 캐시

### 자동 스캔 조정
- `ParkingLotAssetSyncService`는 유지하되 `existsByPartitionKey`로 DB에 이미 있는 키는 건너뜀(현행 동작) → 등록 API와 충돌 없음.
- 영상 업로드는 디스크로 스트리밍 저장(메모리에 통째로 올리지 않음).

## 6. 프론트엔드 UI 흐름 ([app.js](../../../springboot/src/main/resources/static/app.js) 확장)

1. **지도 클릭 → 건물 등록**: 지도에 클릭 리스너 추가. 빈 영역 클릭 시 임시 핀 + 건물 이름 입력 폼(등록/취소).
   등록 → `POST /api/buildings` → 정식 마커로 전환, 목록 갱신.
2. **검색 → 이동**: 검색창 + `naver.maps.Service.geocode()`로 좌표 변환 → `map.setCenter()`. 이후 사용자가 클릭해 핀.
3. **건물 상세 → 주차장 추가**: "주차장 추가" 폼(이름 + 영상 필수 + 사진 선택). 업로드 진행률 표시(영상 큼).
   이후 기존 "사진 업로드 → 지도 제작하기" 흐름으로 슬롯 정의.
4. **삭제**: 건물/주차장 카드에 삭제 버튼(확인 다이얼로그) → `DELETE` → 마커/카드 제거.
5. **권한 노출**: 등록/삭제 UI는 로그인 시에만 표시. 비로그인은 조회만.

별도 페이지 없이 기존 지도+사이드패널 한 화면에서 처리. 기존 카드/폼/fetch 헬퍼 패턴 재사용.

## 7. 영속성 · 인프라 · 권한

- **H2 파일 모드**: [application.properties](../../../springboot/src/main/resources/application.properties)
  `jdbc:h2:mem:testdb` → `jdbc:h2:file:./data/smartparking`. `ddl-auto=update` 유지. `data/`는 `.gitignore` 추가.
- **하드코딩 제거**: `CampusDataInitializer`에서 건물 5개 제거, 캠퍼스 1개만 유지.
  기존 `videos/`에 영상이 있고 건물 매칭이 없으면 자동 스캔은 skip → 필요 시 새로 핀 등록.
- **업로드 한도 상향**: `spring.servlet.multipart.max-file-size`/`max-request-size` `20MB` → `2GB` (영상 700MB~1.4GB 대응).
- **검색 API**: NCP 콘솔에서 Geocoding 활성화(현재 지도 키/계정 그대로, 신규 벤더 가입 불필요).
  소규모/개발 트래픽은 월 무료 한도 내 → 사실상 0원.
- **권한**: 등록/삭제 API는 JWT 필요, 조회 허용. 관리자 역할은 도입하지 않음.

## 8. 테스트 전략

| 대상 | 방식 |
|---|---|
| 건물 생성/삭제 서비스 | 단위 테스트 — 저장, mapKey 고유성, 삭제 시 하위 주차장·파일 정리 |
| 주차장 추가(업로드) | 단위/통합 — partitionKey 자동 생성, 더미 영상 저장 경로 검증(실제 대용량 아님) |
| 신규 엔드포인트 | `@SpringBootTest` + MockMvc — 인증 필요/조회 허용, 200/400/404 |
| 영속성 | H2 파일 모드 저장 후 재조회 검증 |
| 프론트 흐름 | 수동 검증 — 클릭→등록→마커, 검색→이동, 주차장 추가, 삭제 (`run.sh`로 기동) |

FastAPI 쪽 변경 없음 → 테스트 추가 없음.

## 9. 리스크 / 열린 사항

- **대용량 업로드 UX**: 1GB+ 영상 HTTP 업로드는 느림. v1은 진행률 표시로 대응. 추후 청크/직접 배치 고려 가능.
- **Geocoding 정밀도**: 캠퍼스 내부 세부 건물명은 약할 수 있음. 대략 이동 용도로 한정. 추후 POI 검색 확장 여지.
- **슬롯 정의는 여전히 수동**(맵 빌더 GUI). 자동화는 별도 후속 작업.

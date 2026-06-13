#!/usr/bin/env bash
#
# P-Project 통합 실행 스크립트
#   - Spring Boot (웹 + API, 8080) : 백그라운드 실행, 로그는 파일로
#   - FastAPI (YOLO 추론 + 영상 창, 8000) : 포그라운드 실행 (영상 창 표시)
#
# 사용법:  ./run.sh
# 종료:    터미널에서 Ctrl+C  (두 서버 모두 함께 종료됨)
#
set -euo pipefail

# 스크립트 위치를 프로젝트 루트로 사용
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SB_DIR="$ROOT_DIR/springboot"
FA_DIR="$ROOT_DIR/fastapi/video_test"
SB_LOG="/tmp/p-project-springboot.log"

# Java 17 (Homebrew openjdk@17)
export JAVA_HOME="/opt/homebrew/opt/openjdk@17"
export PATH="$JAVA_HOME/bin:$PATH"

# 네이버 지도 키: 이미 환경에 있으면 그대로 쓰고, 없으면 ~/.zshrc 에서 가져옴
if [ -z "${SMARTPARKING_NAVER_MAP_CLIENT_ID:-}" ] && [ -f "$HOME/.zshrc" ]; then
  # shellcheck disable=SC1090
  NAVER_LINE="$(grep -E '^export SMARTPARKING_NAVER_MAP_CLIENT_ID=' "$HOME/.zshrc" | tail -1 || true)"
  [ -n "$NAVER_LINE" ] && eval "$NAVER_LINE"
fi

free_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti:"$port" 2>/dev/null || true)"
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
}

cleanup() {
  echo ""
  echo "[run.sh] 종료 중... 서버 정리"
  free_port 8080
  free_port 8000
  exit 0
}
trap cleanup INT TERM

echo "[run.sh] 기존 포트 정리 (8080, 8000)"
free_port 8080
free_port 8000

# --- Spring Boot (백그라운드) ---
if [ -z "${SMARTPARKING_NAVER_MAP_CLIENT_ID:-}" ]; then
  echo "[run.sh] ⚠️  네이버 지도 키가 없습니다. 지도는 안 뜨지만 나머지는 동작합니다."
else
  echo "[run.sh] 네이버 지도 키 적용됨"
fi
echo "[run.sh] Spring Boot 시작 (로그: $SB_LOG)"
(
  cd "$SB_DIR"
  ./gradlew bootRun
) > "$SB_LOG" 2>&1 &

# Spring Boot 기동 대기
echo -n "[run.sh] Spring Boot 기동 대기"
for _ in $(seq 1 90); do
  if grep -q "Started ServerApplication" "$SB_LOG" 2>/dev/null; then
    echo " -> 완료 (http://localhost:8080/)"
    break
  fi
  echo -n "."
  sleep 1
done

# --- FastAPI (포그라운드, 영상 창 표시) ---
echo "[run.sh] FastAPI(YOLO) 시작 - 영상 창이 뜹니다. 종료하려면 Ctrl+C"
cd "$FA_DIR"
SHOW_GUI=1 ./venv/bin/python server0.py

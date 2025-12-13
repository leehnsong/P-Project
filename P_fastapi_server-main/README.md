# fastapi 실행
vieo_test폴더 -> uvicorn server:app --host 0.0.0.0 --port 8000

# 결과 형식
```
{
  "P1": {
    "occupied_slots": [10, 11, 12, 13, 16, 27],
    "available_slots": [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 17, ...],
    "disabled_slots": [],
    "total_available": 35
  },
  "P2": {
    "occupied_slots": [...],
    "available_slots": [...],
    "disabled_slots": [...],
    "total_available": 0
  },
  "P3": {
    "occupied_slots": [...],
    "available_slots": [...],
    "disabled_slots": [...],
    "total_available": 0
  },
  "last_update": 1733811532.512312
}
```


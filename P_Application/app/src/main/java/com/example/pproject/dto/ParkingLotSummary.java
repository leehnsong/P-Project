package com.example.pproject.dto;

import java.util.List;

public class ParkingLotSummary {
    public Long id;
    public String name;
    public String partitionKey;
    public ParkingLotStatusSummary summary;
    public List<SlotItem> slots; // 건물 상세 응답에서만 채워짐(지도 목록에선 null)
    public String sourceImageUrl; // 예: /api/parking-lots/{id}/map/source-image (없으면 null)
    public Boolean sourceImageExists;
}

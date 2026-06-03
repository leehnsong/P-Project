package com.smartparking.server.dto;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

@Data
public class ParkingStatusResponse {
    
    @JsonProperty("last_update")
    private Double lastUpdate;

    // 1. 기존 partitions 필드는 JSON 결과에서 숨깁니다 (중복 방지)
    @com.fasterxml.jackson.annotation.JsonIgnore
    private Map<String, PartitionData> partitions = new HashMap<>();

    // 2. [데이터 담기] 기존 로직 그대로 유지
    @JsonAnySetter
    public void addPartition(String key, PartitionData value) {
        if (!key.equals("last_update")) {
            this.partitions.put(key, value);
        }
    }

    // 3. [데이터 내보내기] 웹 브라우저로 보낼 때 이 맵의 내용을 펼쳐서 보냄
    @com.fasterxml.jackson.annotation.JsonAnyGetter
    public Map<String, PartitionData> getPartitions() {
        return partitions;
    }
    

    @Data
    public static class PartitionData {
        private SummaryData summary;
        private List<SlotData> slots;
    }

    @Data
    public static class SummaryData {
        private int total;
        private int available;
        @JsonProperty("disabled_available")
        private int disabledAvailable;
    }

    @Data
    public static class SlotData {
        @JsonProperty("slot_id")
        private int slotId;
        private String type;
        private String status;
        private List<Double> center;
    }
}
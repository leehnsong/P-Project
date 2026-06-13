package com.smartparking.server.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ParkingLotSummary {
    private Long id;
    private String name;
    private String partitionKey;
    private ParkingLotStatusSummary summary;
}

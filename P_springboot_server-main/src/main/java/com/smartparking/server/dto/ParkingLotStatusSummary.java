package com.smartparking.server.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ParkingLotStatusSummary {
    private Integer availableSlots;
    private Integer totalSlots;
    private Integer disabledAvailable;
    private Long lastUpdate;
}

package com.example.pproject.dto;

public class ParkingAlertRuleRequest {
    public Long parkingLotId;
    public Integer minimumAvailableSlots;
    public Boolean enabled;

    public ParkingAlertRuleRequest(Long parkingLotId, Integer minimumAvailableSlots, Boolean enabled) {
        this.parkingLotId = parkingLotId;
        this.minimumAvailableSlots = minimumAvailableSlots;
        this.enabled = enabled;
    }
}

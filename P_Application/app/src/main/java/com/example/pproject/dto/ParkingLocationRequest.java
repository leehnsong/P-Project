package com.example.pproject.dto;

public class ParkingLocationRequest {
    public Long parkingLotId;
    public Integer slotId;
    public String vehicleLabel;
    public String memo;

    public ParkingLocationRequest(Long parkingLotId, Integer slotId, String vehicleLabel, String memo) {
        this.parkingLotId = parkingLotId;
        this.slotId = slotId;
        this.vehicleLabel = vehicleLabel;
        this.memo = memo;
    }
}

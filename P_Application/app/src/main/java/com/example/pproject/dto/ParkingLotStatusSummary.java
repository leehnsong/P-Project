package com.example.pproject.dto;

public class ParkingLotStatusSummary {
    public Integer availableSlots;
    public Integer totalSlots;
    public Integer disabledAvailable;
    public Double lastUpdate; // 백엔드가 epoch '초'를 실수(예: 1.78e9)로 보냄. nullable
}

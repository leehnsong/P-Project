package com.smartparking.server.dto;

import lombok.Data;

@Data
public class ParkingStatusResponse {
    private ParkingPartitionStatus P1;
    private ParkingPartitionStatus P2;
    private ParkingPartitionStatus P3;
    private Double last_update;
}
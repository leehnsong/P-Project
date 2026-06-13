package com.smartparking.server.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.List;

@Data
@AllArgsConstructor
public class BuildingDetailResponse {
    private CampusInfo campus;
    private BuildingMapItem building;
    private List<ParkingLotSummary> parkingLots;
}

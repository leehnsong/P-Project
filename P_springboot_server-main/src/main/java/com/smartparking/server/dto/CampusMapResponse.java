package com.smartparking.server.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.List;

@Data
@AllArgsConstructor
public class CampusMapResponse {
    private CampusInfo campus;
    private List<BuildingMapItem> buildings;
}

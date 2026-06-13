package com.smartparking.server.controller;

import com.smartparking.server.dto.BuildingDetailResponse;
import com.smartparking.server.dto.CampusMapResponse;
import com.smartparking.server.dto.UiConfigResponse;
import com.smartparking.server.service.CampusMapService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class CampusMapController {

    private final CampusMapService campusMapService;

    @GetMapping("/api/ui/config")
    public UiConfigResponse getUiConfig() {
        return campusMapService.getUiConfig();
    }

    @GetMapping("/api/campus/map")
    public CampusMapResponse getCampusMap() {
        return campusMapService.getCampusMap();
    }

    @GetMapping("/api/campus/buildings/{buildingId}")
    public ResponseEntity<BuildingDetailResponse> getBuildingDetail(@PathVariable Long buildingId) {
        try {
            return ResponseEntity.ok(campusMapService.getBuildingDetail(buildingId));
        } catch (IllegalArgumentException e) {
            return ResponseEntity.notFound().build();
        }
    }
}

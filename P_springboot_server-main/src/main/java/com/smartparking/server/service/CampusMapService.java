package com.smartparking.server.service;

import com.smartparking.server.dto.BuildingDetailResponse;
import com.smartparking.server.dto.BuildingMapItem;
import com.smartparking.server.dto.CampusInfo;
import com.smartparking.server.dto.CampusMapResponse;
import com.smartparking.server.dto.ParkingLotStatusSummary;
import com.smartparking.server.dto.ParkingLotSummary;
import com.smartparking.server.dto.ParkingPartitionStatus;
import com.smartparking.server.dto.ParkingStatusResponse;
import com.smartparking.server.dto.UiConfigResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class CampusMapService {

    private static final long CAMPUS_ID = 1L;
    private static final long AI_BUILDING_ID = 1L;
    private static final String AI_BUILDING_MAP_KEY = "gachon_ai";

    private final ParkingStatusService parkingStatusService;

    @Value("${smartparking.naver-map.client-id:}")
    private String naverMapClientId;

    @Value("${smartparking.campus.name:Gachon University Global Campus}")
    private String campusName;

    @Value("${smartparking.campus.center-lat:37.4509}")
    private Double campusCenterLat;

    @Value("${smartparking.campus.center-lng:127.1287}")
    private Double campusCenterLng;

    @Value("${smartparking.campus.default-zoom:16}")
    private Integer campusDefaultZoom;

    public UiConfigResponse getUiConfig() {
        return new UiConfigResponse(naverMapClientId, getCampusInfo());
    }

    public CampusMapResponse getCampusMap() {
        return new CampusMapResponse(getCampusInfo(), List.of(getAiBuilding()));
    }

    public BuildingDetailResponse getBuildingDetail(Long buildingId) {
        BuildingMapItem building = getAiBuilding();
        if (buildingId == null || buildingId != AI_BUILDING_ID) {
            throw new IllegalArgumentException("Unknown buildingId: " + buildingId);
        }
        return new BuildingDetailResponse(getCampusInfo(), building, building.getParkingLots());
    }

    private CampusInfo getCampusInfo() {
        return new CampusInfo(
                CAMPUS_ID,
                campusName,
                campusCenterLat,
                campusCenterLng,
                campusDefaultZoom
        );
    }

    private BuildingMapItem getAiBuilding() {
        return new BuildingMapItem(
                AI_BUILDING_ID,
                "AI Engineering Building",
                AI_BUILDING_MAP_KEY,
                campusCenterLat,
                campusCenterLng,
                getParkingLots()
        );
    }

    private List<ParkingLotSummary> getParkingLots() {
        ParkingStatusResponse status = parkingStatusService.getCachedStatus();
        Long lastUpdate = status != null && status.getLast_update() != null
                ? status.getLast_update().longValue()
                : null;

        return List.of(
                new ParkingLotSummary(1L, "P1", "P1", summarize(status != null ? status.getP1() : null, lastUpdate)),
                new ParkingLotSummary(2L, "P2", "P2", summarize(status != null ? status.getP2() : null, lastUpdate)),
                new ParkingLotSummary(3L, "P3", "P3", summarize(status != null ? status.getP3() : null, lastUpdate))
        );
    }

    private ParkingLotStatusSummary summarize(ParkingPartitionStatus partition, Long lastUpdate) {
        if (partition == null) {
            return new ParkingLotStatusSummary(0, 0, 0, lastUpdate);
        }

        int available = partition.getTotal_available() != null
                ? partition.getTotal_available()
                : sizeOf(partition.getAvailable_slots());
        int occupied = sizeOf(partition.getOccupied_slots());
        int disabledAvailable = countDisabledAvailable(partition);

        return new ParkingLotStatusSummary(
                available,
                available + occupied,
                disabledAvailable,
                lastUpdate
        );
    }

    private int sizeOf(List<Integer> values) {
        return values == null ? 0 : values.size();
    }

    private int countDisabledAvailable(ParkingPartitionStatus partition) {
        if (partition.getDisabled_slots() == null || partition.getAvailable_slots() == null) {
            return 0;
        }

        int count = 0;
        for (Integer slot : partition.getDisabled_slots()) {
            if (partition.getAvailable_slots().contains(slot)) {
                count++;
            }
        }
        return count;
    }
}

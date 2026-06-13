package com.example.pproject.dto;

import java.util.List;

public class BuildingMapItem {
    public Long id;
    public String name;
    public String mapKey;
    public Double lat;
    public Double lng;
    public List<ParkingLotSummary> parkingLots;
}

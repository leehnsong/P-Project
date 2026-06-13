package com.smartparking.server.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class CampusInfo {
    private Long id;
    private String name;
    private Double centerLat;
    private Double centerLng;
    private Integer defaultZoom;
}

package com.smartparking.server.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class UiConfigResponse {
    private String naverMapClientId;
    private CampusInfo campus;
}

package com.example.pproject;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.pproject.dto.BuildingMapItem;
import com.example.pproject.dto.CampusInfo;
import com.example.pproject.dto.CampusMapResponse;
import com.example.pproject.dto.ParkingLotSummary;
import com.example.pproject.network.RetrofitClient;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.naver.maps.geometry.LatLng;
import com.naver.maps.map.CameraUpdate;
import com.naver.maps.map.MapFragment;
import com.naver.maps.map.NaverMap;
import com.naver.maps.map.OnMapReadyCallback;
import com.naver.maps.map.overlay.Marker;
import com.naver.maps.map.overlay.Overlay;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class CampusMapActivity extends AppCompatActivity implements OnMapReadyCallback {

    private NaverMap naverMap;
    private TextView tvCampusTitle;
    private TextView tvBuildingSummary;
    private Button btnOpenParkingDetail;
    private FloatingActionButton btnRefreshCampus;
    private final List<Marker> markers = new ArrayList<>();
    private BuildingMapItem selectedBuilding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_campus_map);

        tvCampusTitle = findViewById(R.id.tvCampusTitle);
        tvBuildingSummary = findViewById(R.id.tvBuildingSummary);
        btnOpenParkingDetail = findViewById(R.id.btnOpenParkingDetail);
        btnRefreshCampus = findViewById(R.id.btnRefreshCampus);

        btnOpenParkingDetail.setOnClickListener(v -> {
            Intent intent = new Intent(CampusMapActivity.this, ParkingMapActivity.class);
            startActivity(intent);
        });
        btnRefreshCampus.setOnClickListener(v -> loadCampusMap());

        MapFragment mapFragment = (MapFragment) getSupportFragmentManager()
                .findFragmentById(R.id.naverMapFragment);
        if (mapFragment != null) {
            mapFragment.getMapAsync(this);
        }
    }

    @Override
    public void onMapReady(NaverMap map) {
        naverMap = map;
        naverMap.getUiSettings().setLocationButtonEnabled(false);
        naverMap.getUiSettings().setCompassEnabled(true);
        loadCampusMap();
    }

    private void loadCampusMap() {
        RetrofitClient.getApiService().getCampusMap().enqueue(new Callback<CampusMapResponse>() {
            @Override
            public void onResponse(Call<CampusMapResponse> call, Response<CampusMapResponse> response) {
                if (!response.isSuccessful() || response.body() == null) {
                    showMessage("Campus map request failed: " + response.code());
                    return;
                }

                renderCampusMap(response.body());
            }

            @Override
            public void onFailure(Call<CampusMapResponse> call, Throwable t) {
                showMessage("Campus map network error: " + t.getMessage());
            }
        });
    }

    private void renderCampusMap(CampusMapResponse campusMap) {
        CampusInfo campus = campusMap.campus;
        if (campus != null) {
            tvCampusTitle.setText(campus.name != null ? campus.name : "Campus map");
            if (naverMap != null && campus.centerLat != null && campus.centerLng != null) {
                int zoom = campus.defaultZoom != null ? campus.defaultZoom : 16;
                naverMap.moveCamera(CameraUpdate.scrollAndZoomTo(
                        new LatLng(campus.centerLat, campus.centerLng),
                        zoom
                ));
            }
        }

        clearMarkers();
        List<BuildingMapItem> buildings = campusMap.buildings != null
                ? campusMap.buildings
                : new ArrayList<>();

        for (BuildingMapItem building : buildings) {
            addBuildingMarker(building);
        }

        if (!buildings.isEmpty()) {
            selectBuilding(buildings.get(0), false);
        } else {
            selectedBuilding = null;
            btnOpenParkingDetail.setEnabled(false);
            tvBuildingSummary.setText("No parking buildings were found.");
        }
    }

    private void addBuildingMarker(BuildingMapItem building) {
        if (naverMap == null || building.lat == null || building.lng == null) {
            return;
        }

        Marker marker = new Marker();
        marker.setPosition(new LatLng(building.lat, building.lng));
        marker.setCaptionText(building.name != null ? building.name : "Building");
        marker.setSubCaptionText(building.mapKey != null ? building.mapKey : "");
        marker.setMap(naverMap);
        marker.setOnClickListener((Overlay overlay) -> {
            selectBuilding(building, true);
            return true;
        });
        markers.add(marker);
    }

    private void selectBuilding(BuildingMapItem building, boolean moveCamera) {
        selectedBuilding = building;
        btnOpenParkingDetail.setEnabled(true);
        tvBuildingSummary.setText(buildSummaryText(building));

        if (moveCamera && naverMap != null && building.lat != null && building.lng != null) {
            naverMap.moveCamera(CameraUpdate.scrollTo(new LatLng(building.lat, building.lng)));
        }
    }

    private String buildSummaryText(BuildingMapItem building) {
        int totalAvailable = 0;
        int totalSlots = 0;
        int lotCount = 0;

        if (building.parkingLots != null) {
            lotCount = building.parkingLots.size();
            for (ParkingLotSummary lot : building.parkingLots) {
                if (lot != null && lot.summary != null) {
                    totalAvailable += valueOrZero(lot.summary.availableSlots);
                    totalSlots += valueOrZero(lot.summary.totalSlots);
                }
            }
        }

        return String.format(
                Locale.KOREA,
                "%s\nParking lots: %d\nAvailable: %d / %d",
                building.name != null ? building.name : "Selected building",
                lotCount,
                totalAvailable,
                totalSlots
        );
    }

    private int valueOrZero(Integer value) {
        return value == null ? 0 : value;
    }

    private void clearMarkers() {
        for (Marker marker : markers) {
            marker.setMap(null);
        }
        markers.clear();
    }

    private void showMessage(String message) {
        tvBuildingSummary.setText(message);
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
    }
}

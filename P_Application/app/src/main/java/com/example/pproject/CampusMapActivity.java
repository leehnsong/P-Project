package com.example.pproject;

import android.content.Intent;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.example.pproject.dto.GeoSearchResult;
import com.example.pproject.dto.ParkingLocationResponse;

import androidx.appcompat.app.AppCompatActivity;

import com.example.pproject.dto.BuildingMapItem;
import com.example.pproject.dto.CampusInfo;
import com.example.pproject.dto.CampusMapResponse;
import com.example.pproject.dto.ParkingLotStatusSummary;
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
    private LinearLayout lotListContainer;
    private FloatingActionButton btnRefreshCampus;
    private EditText etSearch;
    private Button btnSearch;
    private TextView tvMapMyParking;
    private final List<Marker> markers = new ArrayList<>();
    private BuildingMapItem selectedBuilding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_campus_map);

        tvCampusTitle = findViewById(R.id.tvCampusTitle);
        tvBuildingSummary = findViewById(R.id.tvBuildingSummary);
        lotListContainer = findViewById(R.id.lotListContainer);
        btnRefreshCampus = findViewById(R.id.btnRefreshCampus);
        etSearch = findViewById(R.id.etSearch);
        btnSearch = findViewById(R.id.btnSearch);
        tvMapMyParking = findViewById(R.id.tvMapMyParking);

        btnRefreshCampus.setOnClickListener(v -> loadCampusMap());
        btnSearch.setOnClickListener(v -> runSearch());
        etSearch.setOnEditorActionListener((tv, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_SEARCH) {
                runSearch();
                return true;
            }
            return false;
        });

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
                    showMessage("주차장 지도를 불러오지 못했습니다 (" + response.code() + ")");
                    return;
                }
                renderCampusMap(response.body());
            }

            @Override
            public void onFailure(Call<CampusMapResponse> call, Throwable t) {
                showMessage("네트워크 오류: 백엔드 서버가 켜져 있는지 확인해 주세요");
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadMyParking();
    }

    private void loadMyParking() {
        if (!TokenStore.isLoggedIn(this)) {
            tvMapMyParking.setVisibility(View.GONE);
            return;
        }
        RetrofitClient.getApiService().getMyParkingLocation()
                .enqueue(new Callback<ParkingLocationResponse>() {
                    @Override
                    public void onResponse(Call<ParkingLocationResponse> call,
                                           Response<ParkingLocationResponse> response) {
                        ParkingLocationResponse loc = response.isSuccessful() ? response.body() : null;
                        if (loc != null && loc.active) {
                            String name = loc.parkingLotName != null ? loc.parkingLotName : "주차장";
                            tvMapMyParking.setText(String.format(Locale.KOREA,
                                    "🚗 내 주차 위치: %s %d번", name, loc.slotId));
                            tvMapMyParking.setVisibility(View.VISIBLE);
                        } else {
                            tvMapMyParking.setVisibility(View.GONE);
                        }
                    }

                    @Override
                    public void onFailure(Call<ParkingLocationResponse> call, Throwable t) {
                        tvMapMyParking.setVisibility(View.GONE);
                    }
                });
    }

    private void runSearch() {
        String query = etSearch.getText() != null ? etSearch.getText().toString().trim() : "";
        if (query.isEmpty()) {
            return;
        }
        RetrofitClient.getApiService().searchPlaces(query).enqueue(new Callback<List<GeoSearchResult>>() {
            @Override
            public void onResponse(Call<List<GeoSearchResult>> call, Response<List<GeoSearchResult>> response) {
                if (!response.isSuccessful() || response.body() == null || response.body().isEmpty()) {
                    Toast.makeText(CampusMapActivity.this, "검색 결과가 없습니다.", Toast.LENGTH_SHORT).show();
                    return;
                }
                GeoSearchResult top = response.body().get(0);
                if (naverMap != null) {
                    naverMap.moveCamera(CameraUpdate.scrollAndZoomTo(new LatLng(top.lat, top.lng), 17));
                }
            }

            @Override
            public void onFailure(Call<List<GeoSearchResult>> call, Throwable t) {
                Toast.makeText(CampusMapActivity.this, "검색 실패: 백엔드 서버를 확인해 주세요", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void renderCampusMap(CampusMapResponse campusMap) {
        CampusInfo campus = campusMap.campus;
        if (campus != null) {
            tvCampusTitle.setText(campus.name != null ? campus.name : "캠퍼스 지도");
            if (naverMap != null && campus.centerLat != null && campus.centerLng != null) {
                int zoom = campus.defaultZoom != null ? campus.defaultZoom : 16;
                naverMap.moveCamera(CameraUpdate.scrollAndZoomTo(
                        new LatLng(campus.centerLat, campus.centerLng), zoom));
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
            tvBuildingSummary.setText("등록된 주차장이 없습니다.");
            lotListContainer.removeAllViews();
        }
    }

    private void addBuildingMarker(BuildingMapItem building) {
        if (naverMap == null || building.lat == null || building.lng == null) {
            return;
        }

        Marker marker = new Marker();
        marker.setPosition(new LatLng(building.lat, building.lng));
        marker.setCaptionText(building.name != null ? building.name : "건물");
        marker.setMap(naverMap);
        marker.setOnClickListener((Overlay overlay) -> {
            selectBuilding(building, true);
            return true;
        });
        markers.add(marker);
    }

    private void selectBuilding(BuildingMapItem building, boolean moveCamera) {
        selectedBuilding = building;

        int lotCount = building.parkingLots != null ? building.parkingLots.size() : 0;
        String name = building.name != null ? building.name : "선택한 건물";
        tvBuildingSummary.setText(String.format(Locale.KOREA, "%s · 주차장 %d곳", name, lotCount));

        renderLotCards(building);

        if (moveCamera && naverMap != null && building.lat != null && building.lng != null) {
            naverMap.moveCamera(CameraUpdate.scrollTo(new LatLng(building.lat, building.lng)));
        }
    }

    private void renderLotCards(BuildingMapItem building) {
        lotListContainer.removeAllViews();
        List<ParkingLotSummary> lots = building.parkingLots;
        if (lots == null || lots.isEmpty()) {
            TextView empty = new TextView(this);
            empty.setText("이 건물에는 등록된 주차장이 없습니다.");
            empty.setTextColor(0xFF757575);
            lotListContainer.addView(empty);
            return;
        }
        Long buildingId = building.id;
        for (ParkingLotSummary lot : lots) {
            lotListContainer.addView(createLotRow(buildingId, lot));
        }
    }

    private View createLotRow(Long buildingId, ParkingLotSummary lot) {
        View row = getLayoutInflater().inflate(R.layout.item_lot_row, lotListContainer, false);
        TextView tvName = row.findViewById(R.id.tvLotName);
        TextView tvCount = row.findViewById(R.id.tvLotCount);
        TextView tvStatus = row.findViewById(R.id.tvLotStatus);

        tvName.setText(lot.name != null ? lot.name : "주차장");

        ParkingLotStatusSummary s = lot.summary;
        int total = s != null ? valueOrZero(s.totalSlots) : 0;
        int avail = s != null ? valueOrZero(s.availableSlots) : 0;

        String label;
        int color;
        if (s == null || total == 0) {
            label = "정보없음";
            color = 0xFF9E9E9E;
            tvCount.setText("점유 정보 없음");
        } else {
            if (avail == 0) {
                label = "만차";
                color = 0xFFF44336;
            } else if (avail <= 4) {
                label = "혼잡";
                color = 0xFFFF9800;
            } else {
                label = "여유";
                color = 0xFF4CAF50;
            }
            tvCount.setText(String.format(Locale.KOREA, "빈자리 %d / 총 %d칸", avail, total));
        }

        tvStatus.setText(label);
        GradientDrawable bg = new GradientDrawable();
        bg.setColor(color);
        bg.setCornerRadius(dp(14));
        tvStatus.setBackground(bg);

        // 카드 클릭 → 슬롯 격자 화면으로 이동
        if (buildingId != null && lot.id != null) {
            row.setOnClickListener(v -> {
                Intent intent = new Intent(CampusMapActivity.this, SlotGridActivity.class);
                intent.putExtra(SlotGridActivity.EXTRA_BUILDING_ID, buildingId);
                intent.putExtra(SlotGridActivity.EXTRA_PARKING_LOT_ID, lot.id);
                intent.putExtra(SlotGridActivity.EXTRA_LOT_NAME, lot.name);
                startActivity(intent);
            });
        }

        // ☆ 즐겨찾기 토글
        TextView tvStar = row.findViewById(R.id.tvFavStar);
        if (lot.id != null) {
            updateStar(tvStar, FavoriteStore.isFavorite(this, lot.id));
            tvStar.setOnClickListener(v -> {
                FavoriteStore.toggle(this, lot.id);
                updateStar(tvStar, FavoriteStore.isFavorite(this, lot.id));
            });
        } else {
            tvStar.setVisibility(View.GONE);
        }

        return row;
    }

    private void updateStar(TextView star, boolean on) {
        star.setText(on ? "★" : "☆");
    }

    private float dp(float value) {
        return value * getResources().getDisplayMetrics().density;
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

package com.example.pproject;

import android.content.res.ColorStateList;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.activity.OnBackPressedCallback; // [추가됨]
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.view.ViewCompat;

import com.example.pproject.dto.ParkingPartitionStatus;
import com.example.pproject.dto.ParkingStatusResponse;
import com.example.pproject.network.RetrofitClient;
import com.google.android.material.floatingactionbutton.FloatingActionButton;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class ParkingMapActivity extends AppCompatActivity {

    FloatingActionButton btnRefresh;
    Button btnTab1, btnTab2, btnTab3;
    View layoutP1, layoutP2, layoutP3, layoutOverview;
    View tabLayout;

    Button btnZone1Main, btnZone2Main, btnZone3Main;

    TextView tvAvailableCount, tvSubtitle;

    private ParkingStatusResponse currentData;

    // 0: 전체 약도, 1~3: 상세 구역
    private int currentZone = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_parking_map);

        initViews();
        setupListeners();
        setupBackPressHandler(); // [추가됨] 뒤로가기 핸들러 등록

        // 초기 상태: 전체 약도
        switchZone(0);
        loadParkingStatus();
    }

    // 뒤로가기 처리
    private void setupBackPressHandler() {
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                // 상세 구역을 보고 있다면 -> 전체 약도로 돌아감
                if (currentZone != 0) {
                    switchZone(0);
                } else {
                    // 이미 전체 약도라면 -> 기본 동작(앱 종료/이전 화면) 수행
                    setEnabled(false); // 이 콜백을 비활성화하고
                    getOnBackPressedDispatcher().onBackPressed(); // 다시 뒤로가기를 호출하여 기본 동작 실행
                }
            }
        });
    }

    private void initViews() {
        btnRefresh = findViewById(R.id.btnRefresh);
        tabLayout = findViewById(R.id.tabLayout);

        btnTab1 = findViewById(R.id.btnTab1);
        btnTab2 = findViewById(R.id.btnTab2);
        btnTab3 = findViewById(R.id.btnTab3);

        tvAvailableCount = findViewById(R.id.tvAvailableCount);
        tvSubtitle = findViewById(R.id.tvSubtitle);

        layoutP1 = findViewById(R.id.includeP1);
        layoutP2 = findViewById(R.id.includeP2);
        layoutP3 = findViewById(R.id.includeP3);
        layoutOverview = findViewById(R.id.includeOverview);

        btnZone1Main = findViewById(R.id.btnZone1Main);
        btnZone2Main = findViewById(R.id.btnZone2Main);
        btnZone3Main = findViewById(R.id.btnZone3Main);
    }

    private void setupListeners() {
        btnTab1.setOnClickListener(v -> switchZone(1));
        btnTab2.setOnClickListener(v -> switchZone(2));
        btnTab3.setOnClickListener(v -> switchZone(3));

        btnZone1Main.setOnClickListener(v -> switchZone(1));
        btnZone2Main.setOnClickListener(v -> switchZone(2));
        btnZone3Main.setOnClickListener(v -> switchZone(3));

        btnRefresh.setOnClickListener(v -> loadParkingStatus());
    }

    private void switchZone(int zone) {
        this.currentZone = zone;

        if (zone == 0) {
            // [전체 약도 모드]
            layoutOverview.setVisibility(View.VISIBLE);
            tabLayout.setVisibility(View.GONE);

            layoutP1.setVisibility(View.GONE);
            layoutP2.setVisibility(View.GONE);
            layoutP3.setVisibility(View.GONE);

            if (currentData != null) {
                int total = getTotalAvailable(currentData.P1) + getTotalAvailable(currentData.P2) + getTotalAvailable(currentData.P3);
                tvAvailableCount.setText(total + " 대");
                tvSubtitle.setText("전체 주차장 현황");
            } else {
                tvAvailableCount.setText("- 대");
            }

        } else {
            // [상세 지도 모드]
            layoutOverview.setVisibility(View.GONE);
            tabLayout.setVisibility(View.VISIBLE);

            layoutP1.setVisibility(zone == 1 ? View.VISIBLE : View.GONE);
            layoutP2.setVisibility(zone == 2 ? View.VISIBLE : View.GONE);
            layoutP3.setVisibility(zone == 3 ? View.VISIBLE : View.GONE);

            int colorGray = ContextCompat.getColor(this, android.R.color.darker_gray);
            int colorPrimary = ContextCompat.getColor(this, R.color.brand_primary);

            setButtonColor(btnTab1, zone == 1 ? colorPrimary : colorGray);
            setButtonColor(btnTab2, zone == 2 ? colorPrimary : colorGray);
            setButtonColor(btnTab3, zone == 3 ? colorPrimary : colorGray);

            updateHeaderUI();
            tvSubtitle.setText("*장애인 주차구역 포함");
        }
    }

    private void setButtonColor(Button btn, int color) {
        ViewCompat.setBackgroundTintList(btn, ColorStateList.valueOf(color));
    }

    private void loadParkingStatus() {
        RetrofitClient.getApiService().getParkingStatus().enqueue(new Callback<ParkingStatusResponse>() {
            @Override
            public void onResponse(Call<ParkingStatusResponse> call, Response<ParkingStatusResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    currentData = response.body();

                    applyColorsToPartition(currentData.P1);
                    applyColorsToPartition(currentData.P2);
                    applyColorsToPartition(currentData.P3);

                    if (currentZone == 0) {
                        int total = getTotalAvailable(currentData.P1) + getTotalAvailable(currentData.P2) + getTotalAvailable(currentData.P3);
                        tvAvailableCount.setText(total + " 대");
                    } else {
                        updateHeaderUI();
                    }
                }
            }

            @Override
            public void onFailure(Call<ParkingStatusResponse> call, Throwable t) {
                Log.e("API", "Failed: " + t.getMessage());
            }
        });
    }

    private void updateHeaderUI() {
        if (currentData == null) return;

        ParkingPartitionStatus targetStatus = null;
        if (currentZone == 1) targetStatus = currentData.P1;
        else if (currentZone == 2) targetStatus = currentData.P2;
        else if (currentZone == 3) targetStatus = currentData.P3;

        if (targetStatus != null) {
            tvAvailableCount.setText(getTotalAvailable(targetStatus) + " 대");
        }
    }

    private int getTotalAvailable(ParkingPartitionStatus p) {
        if (p == null) return 0;
        if (p.total_available != null) return p.total_available;
        if (p.available_slots != null) return p.available_slots.size();
        return 0;
    }

    private void applyColorsToPartition(ParkingPartitionStatus p) {
        if (p == null) return;

        if (p.occupied_slots != null) {
            for (int slot : p.occupied_slots) setSlotColor(slot, R.color.parking_occupied);
        }
        if (p.available_slots != null) {
            for (int slot : p.available_slots) setSlotColor(slot, R.color.parking_available);
        }
        if (p.disabled_slots != null) {
            for (int slot : p.disabled_slots) {
                if (p.available_slots != null && p.available_slots.contains(slot)) {
                    setSlotColor(slot, R.color.parking_disabled);
                }
            }
        }
    }

    private void setSlotColor(int slot, int colorRes) {
        String viewId = "slot" + slot;
        int id = getResources().getIdentifier(viewId, "id", getPackageName());
        View v = findViewById(id);
        if (v != null) {
            v.setBackgroundColor(ContextCompat.getColor(this, colorRes));
        }
    }
}

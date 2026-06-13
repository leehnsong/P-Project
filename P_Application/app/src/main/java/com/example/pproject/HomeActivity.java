package com.example.pproject;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

import android.widget.Button;
import android.widget.Toast;

import com.example.pproject.dto.BuildingMapItem;
import com.example.pproject.dto.CampusMapResponse;
import com.example.pproject.dto.ParkingLocationResponse;
import com.example.pproject.dto.ParkingLotStatusSummary;
import com.example.pproject.dto.ParkingLotSummary;
import com.example.pproject.network.RetrofitClient;

import java.util.Locale;
import java.util.Set;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class HomeActivity extends AppCompatActivity {

    private View btnGoToMap, btnGoToVoice;
    private LinearLayout favListContainer;
    private View myParkingBanner;
    private TextView tvMyParking;
    private Button btnHomeRelease;
    private View btnGoToNotifications;
    private TextView tvNotifLabel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);

        btnGoToMap = findViewById(R.id.btnGoToMap);
        btnGoToVoice = findViewById(R.id.btnGoToVoice);
        favListContainer = findViewById(R.id.favListContainer);
        myParkingBanner = findViewById(R.id.myParkingBanner);
        tvMyParking = findViewById(R.id.tvMyParking);
        btnHomeRelease = findViewById(R.id.btnHomeRelease);
        btnGoToNotifications = findViewById(R.id.btnGoToNotifications);
        tvNotifLabel = findViewById(R.id.tvNotifLabel);

        btnGoToNotifications.setOnClickListener(v ->
                startActivity(new Intent(HomeActivity.this, NotificationsActivity.class)));

        btnGoToMap.setOnClickListener(v ->
                startActivity(new Intent(HomeActivity.this, CampusMapActivity.class)));

        btnGoToVoice.setOnClickListener(v ->
                startActivity(new Intent(HomeActivity.this, VoiceActivity.class)));

        btnHomeRelease.setOnClickListener(v -> releaseMyParking());
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadFavorites();   // 즐겨찾기 토글 후 돌아올 때마다 갱신
        loadMyParking();   // 내 주차 위치 배너 갱신
        loadUnreadCount(); // 알림함 안읽음 개수 갱신
    }

    private void loadUnreadCount() {
        if (!TokenStore.isLoggedIn(this)) {
            tvNotifLabel.setText("🔔 알림함");
            return;
        }
        RetrofitClient.getApiService().getUnreadCount()
                .enqueue(new retrofit2.Callback<com.example.pproject.dto.UnreadCountResponse>() {
                    @Override
                    public void onResponse(retrofit2.Call<com.example.pproject.dto.UnreadCountResponse> call,
                                           retrofit2.Response<com.example.pproject.dto.UnreadCountResponse> response) {
                        long n = response.isSuccessful() && response.body() != null
                                ? response.body().unreadCount : 0;
                        tvNotifLabel.setText(n > 0 ? "🔔 알림함 (" + n + ")" : "🔔 알림함");
                    }

                    @Override
                    public void onFailure(retrofit2.Call<com.example.pproject.dto.UnreadCountResponse> call, Throwable t) {
                        tvNotifLabel.setText("🔔 알림함");
                    }
                });
    }

    private void loadMyParking() {
        if (!TokenStore.isLoggedIn(this)) {
            myParkingBanner.setVisibility(View.GONE);
            return;
        }
        RetrofitClient.getApiService().getMyParkingLocation()
                .enqueue(new retrofit2.Callback<ParkingLocationResponse>() {
                    @Override
                    public void onResponse(retrofit2.Call<ParkingLocationResponse> call,
                                           retrofit2.Response<ParkingLocationResponse> response) {
                        ParkingLocationResponse loc = response.isSuccessful() ? response.body() : null;
                        if (loc != null && loc.active) {
                            String name = loc.parkingLotName != null ? loc.parkingLotName : "주차장";
                            tvMyParking.setText(String.format(Locale.KOREA,
                                    "🚗 내 주차 위치: %s %d번", name, loc.slotId));
                            myParkingBanner.setVisibility(View.VISIBLE);
                        } else {
                            myParkingBanner.setVisibility(View.GONE);
                        }
                    }

                    @Override
                    public void onFailure(retrofit2.Call<ParkingLocationResponse> call, Throwable t) {
                        myParkingBanner.setVisibility(View.GONE);
                    }
                });
    }

    private void releaseMyParking() {
        RetrofitClient.getApiService().releaseMyParkingLocation()
                .enqueue(new retrofit2.Callback<ParkingLocationResponse>() {
                    @Override
                    public void onResponse(retrofit2.Call<ParkingLocationResponse> call,
                                           retrofit2.Response<ParkingLocationResponse> response) {
                        myParkingBanner.setVisibility(View.GONE);
                        Toast.makeText(HomeActivity.this, "주차를 종료했어요.", Toast.LENGTH_SHORT).show();
                    }

                    @Override
                    public void onFailure(retrofit2.Call<ParkingLocationResponse> call, Throwable t) {
                        Toast.makeText(HomeActivity.this, "네트워크 오류로 해제하지 못했어요.", Toast.LENGTH_SHORT).show();
                    }
                });
    }

    private void loadFavorites() {
        Set<String> favIds = FavoriteStore.getIds(this);
        favListContainer.removeAllViews();

        if (favIds.isEmpty()) {
            showFavMessage("아직 즐겨찾기한 주차장이 없습니다.\n지도에서 ☆를 눌러 추가해보세요.");
            return;
        }

        RetrofitClient.getApiService().getCampusMap().enqueue(new Callback<CampusMapResponse>() {
            @Override
            public void onResponse(Call<CampusMapResponse> call, Response<CampusMapResponse> response) {
                if (!response.isSuccessful() || response.body() == null) {
                    showFavMessage("주차장 정보를 불러오지 못했습니다.");
                    return;
                }
                renderFavorites(response.body(), FavoriteStore.getIds(HomeActivity.this));
            }

            @Override
            public void onFailure(Call<CampusMapResponse> call, Throwable t) {
                showFavMessage("네트워크 오류: 백엔드 서버를 확인해 주세요.");
            }
        });
    }

    private void renderFavorites(CampusMapResponse campusMap, Set<String> favIds) {
        favListContainer.removeAllViews();
        int shown = 0;

        if (campusMap.buildings != null) {
            for (BuildingMapItem building : campusMap.buildings) {
                if (building.parkingLots == null) {
                    continue;
                }
                for (ParkingLotSummary lot : building.parkingLots) {
                    if (lot.id != null && favIds.contains(String.valueOf(lot.id))) {
                        favListContainer.addView(createFavRow(building.id, lot));
                        shown++;
                    }
                }
            }
        }

        if (shown == 0) {
            showFavMessage("즐겨찾기한 주차장을 찾을 수 없습니다.");
        }
    }

    private View createFavRow(Long buildingId, ParkingLotSummary lot) {
        View row = getLayoutInflater().inflate(R.layout.item_lot_row, favListContainer, false);
        TextView tvName = row.findViewById(R.id.tvLotName);
        TextView tvCount = row.findViewById(R.id.tvLotCount);
        TextView tvStatus = row.findViewById(R.id.tvLotStatus);
        TextView tvStar = row.findViewById(R.id.tvFavStar);

        tvName.setText(lot.name != null ? lot.name : "주차장");

        ParkingLotStatusSummary s = lot.summary;
        int total = s != null && s.totalSlots != null ? s.totalSlots : 0;
        int avail = s != null && s.availableSlots != null ? s.availableSlots : 0;

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
        bg.setCornerRadius(14f * getResources().getDisplayMetrics().density);
        tvStatus.setBackground(bg);

        // 즐겨찾기 화면이므로 별은 채워진 상태, 누르면 해제
        tvStar.setText("★");
        if (lot.id != null) {
            tvStar.setOnClickListener(v -> {
                FavoriteStore.toggle(this, lot.id);
                loadFavorites();
            });
            if (buildingId != null) {
                row.setOnClickListener(v -> {
                    Intent intent = new Intent(HomeActivity.this, SlotGridActivity.class);
                    intent.putExtra(SlotGridActivity.EXTRA_BUILDING_ID, buildingId);
                    intent.putExtra(SlotGridActivity.EXTRA_PARKING_LOT_ID, lot.id);
                    intent.putExtra(SlotGridActivity.EXTRA_LOT_NAME, lot.name);
                    startActivity(intent);
                });
            }
        }

        return row;
    }

    private void showFavMessage(String message) {
        favListContainer.removeAllViews();
        TextView tv = new TextView(this);
        tv.setText(message);
        tv.setTextColor(0xFF9E9E9E);
        tv.setTextSize(14);
        favListContainer.addView(tv);
    }
}

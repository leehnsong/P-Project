package com.example.pproject;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.text.InputType;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.example.pproject.dto.BuildingDetailResponse;
import com.example.pproject.dto.ParkingAlertRuleRequest;
import com.example.pproject.dto.ParkingAlertRuleResponse;
import com.example.pproject.dto.ParkingLocationRequest;
import com.example.pproject.dto.ParkingLocationResponse;
import com.example.pproject.dto.ParkingLotStatusSummary;
import com.example.pproject.dto.ParkingLotSummary;
import com.example.pproject.dto.SlotItem;
import com.example.pproject.network.RetrofitClient;

import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.List;
import java.util.Locale;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

/**
 * 한 주차장(파티션)의 원본 사진 위에 슬롯을 실제 위치대로 색상 오버레이로 표시.
 * 건물 상세(/api/campus/buildings/{id})에서 해당 주차장의 sourceImageUrl + slots를 사용한다.
 * (Phase 2에서 슬롯 탭 → 내 주차위치 저장으로 확장 예정)
 */
public class SlotGridActivity extends AppCompatActivity {

    public static final String EXTRA_BUILDING_ID = "buildingId";
    public static final String EXTRA_PARKING_LOT_ID = "parkingLotId";
    public static final String EXTRA_LOT_NAME = "lotName";

    private TextView tvSlotTitle;
    private TextView tvSlotSummary;
    private SlotMapView slotMap;
    private View myLocBar;
    private TextView tvMyLoc;
    private Button btnRelease;
    private Button btnAlert;

    private long buildingId;
    private long parkingLotId;
    private String lotName;

    private Long alertRuleId;          // 이 주차장에 등록된 알림 규칙 id (없으면 null)
    private Integer alertThreshold;    // 등록된 임계값

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_slot_grid);

        tvSlotTitle = findViewById(R.id.tvSlotTitle);
        tvSlotSummary = findViewById(R.id.tvSlotSummary);
        slotMap = findViewById(R.id.slotMap);
        myLocBar = findViewById(R.id.myLocBar);
        tvMyLoc = findViewById(R.id.tvMyLoc);
        btnRelease = findViewById(R.id.btnRelease);
        btnAlert = findViewById(R.id.btnAlert);

        buildingId = getIntent().getLongExtra(EXTRA_BUILDING_ID, -1);
        parkingLotId = getIntent().getLongExtra(EXTRA_PARKING_LOT_ID, -1);
        lotName = getIntent().getStringExtra(EXTRA_LOT_NAME);
        tvSlotTitle.setText(lotName != null ? lotName : "주차장");

        slotMap.setOnSlotClickListener(this::onSlotTapped);
        btnRelease.setOnClickListener(v -> releaseLocation());
        btnAlert.setOnClickListener(v -> showAlertDialog());

        loadSlots();
        loadMyLocation();
        loadAlertRule();
    }

    private void loadSlots() {
        if (buildingId < 0 || parkingLotId < 0) {
            tvSlotSummary.setText("잘못된 접근입니다.");
            return;
        }
        RetrofitClient.getApiService().getBuildingDetail(buildingId)
                .enqueue(new Callback<BuildingDetailResponse>() {
                    @Override
                    public void onResponse(Call<BuildingDetailResponse> call,
                                           Response<BuildingDetailResponse> response) {
                        if (!response.isSuccessful() || response.body() == null) {
                            tvSlotSummary.setText("정보를 불러오지 못했습니다 (" + response.code() + ")");
                            return;
                        }
                        ParkingLotSummary lot = findLot(response.body());
                        if (lot == null) {
                            tvSlotSummary.setText("주차장을 찾을 수 없습니다.");
                            return;
                        }
                        renderLot(lot);
                    }

                    @Override
                    public void onFailure(Call<BuildingDetailResponse> call, Throwable t) {
                        tvSlotSummary.setText("네트워크 오류: 백엔드 서버를 확인해 주세요");
                    }
                });
    }

    private ParkingLotSummary findLot(BuildingDetailResponse body) {
        if (body.parkingLots == null) {
            return null;
        }
        for (ParkingLotSummary lot : body.parkingLots) {
            if (lot != null && lot.id != null && lot.id == parkingLotId) {
                return lot;
            }
        }
        return null;
    }

    private void renderLot(ParkingLotSummary lot) {
        ParkingLotStatusSummary s = lot.summary;
        if (s != null && s.totalSlots != null) {
            int avail = s.availableSlots != null ? s.availableSlots : 0;
            tvSlotSummary.setText(String.format(Locale.KOREA, "빈자리 %d / 총 %d칸", avail, s.totalSlots));
        } else {
            tvSlotSummary.setText("점유 정보 없음");
        }

        // 슬롯 마커 먼저 표시(사진은 로드되면 배경으로 깔림)
        slotMap.setData(null, lot.slots);

        if (lot.slots == null || lot.slots.isEmpty()) {
            tvSlotSummary.append("\n(슬롯이 정의되지 않았습니다. 웹 맵 빌더로 슬롯을 먼저 그려주세요.)");
        }

        if (lot.sourceImageUrl != null && !lot.sourceImageUrl.isEmpty()) {
            loadImageAsync(RetrofitClient.getServerOrigin() + lot.sourceImageUrl, lot);
        } else {
            tvSlotSummary.append("\n(원본 사진이 없어 위치만 표시합니다. 웹에서 사진을 업로드하세요.)");
        }
    }

    // ===== 빈자리 알림 (Phase 3) =====

    private void loadAlertRule() {
        if (!TokenStore.isLoggedIn(this)) {
            return;
        }
        RetrofitClient.getApiService().getAlertRules()
                .enqueue(new Callback<List<ParkingAlertRuleResponse>>() {
                    @Override
                    public void onResponse(Call<List<ParkingAlertRuleResponse>> call,
                                           Response<List<ParkingAlertRuleResponse>> response) {
                        alertRuleId = null;
                        alertThreshold = null;
                        if (response.isSuccessful() && response.body() != null) {
                            for (ParkingAlertRuleResponse rule : response.body()) {
                                if (rule.parkingLotId != null && rule.parkingLotId == parkingLotId && rule.enabled) {
                                    alertRuleId = rule.id;
                                    alertThreshold = rule.minimumAvailableSlots;
                                    break;
                                }
                            }
                        }
                        updateAlertButton();
                    }

                    @Override
                    public void onFailure(Call<List<ParkingAlertRuleResponse>> call, Throwable t) {
                        // 무시
                    }
                });
    }

    private void updateAlertButton() {
        btnAlert.setText(alertRuleId != null ? "🔔 알림 켜짐" : "🔔 알림");
    }

    private void showAlertDialog() {
        if (!TokenStore.isLoggedIn(this)) {
            Toast.makeText(this, "빈자리 알림은 로그인 후 이용할 수 있어요.", Toast.LENGTH_SHORT).show();
            return;
        }

        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_NUMBER);
        input.setHint("빈자리 개수");
        input.setText(String.valueOf(alertThreshold != null ? alertThreshold : 1));

        AlertDialog.Builder b = new AlertDialog.Builder(this)
                .setTitle("빈자리 알림")
                .setMessage("빈자리가 몇 개 이상일 때 알림을 받을까요?")
                .setView(input)
                .setPositiveButton(alertRuleId != null ? "변경" : "등록", (d, w) -> {
                    int n;
                    try {
                        n = Integer.parseInt(input.getText().toString().trim());
                    } catch (NumberFormatException e) {
                        n = 1;
                    }
                    if (n < 1) {
                        n = 1;
                    }
                    registerAlert(n);
                })
                .setNegativeButton("취소", null);

        if (alertRuleId != null) {
            b.setNeutralButton("해제", (d, w) -> deleteAlert());
        }
        b.show();
    }

    private void registerAlert(int threshold) {
        // 기존 규칙이 있으면 삭제 후 새로 등록(중복 방지)
        if (alertRuleId != null) {
            RetrofitClient.getApiService().deleteAlertRule(alertRuleId)
                    .enqueue(new Callback<Void>() {
                        @Override
                        public void onResponse(Call<Void> call, Response<Void> response) {
                            createAlert(threshold);
                        }

                        @Override
                        public void onFailure(Call<Void> call, Throwable t) {
                            createAlert(threshold);
                        }
                    });
        } else {
            createAlert(threshold);
        }
    }

    private void createAlert(int threshold) {
        ParkingAlertRuleRequest req =
                new ParkingAlertRuleRequest(parkingLotId, threshold, true);
        RetrofitClient.getApiService().createAlertRule(req)
                .enqueue(new Callback<ParkingAlertRuleResponse>() {
                    @Override
                    public void onResponse(Call<ParkingAlertRuleResponse> call,
                                           Response<ParkingAlertRuleResponse> response) {
                        if (response.isSuccessful() && response.body() != null) {
                            alertRuleId = response.body().id;
                            alertThreshold = response.body().minimumAvailableSlots;
                            updateAlertButton();
                            Toast.makeText(SlotGridActivity.this,
                                    "빈자리 " + threshold + "개 이상이면 알림을 보낼게요.", Toast.LENGTH_SHORT).show();
                        } else {
                            Toast.makeText(SlotGridActivity.this, "알림 등록 실패 (" + response.code() + ")", Toast.LENGTH_SHORT).show();
                        }
                    }

                    @Override
                    public void onFailure(Call<ParkingAlertRuleResponse> call, Throwable t) {
                        Toast.makeText(SlotGridActivity.this, "네트워크 오류로 등록하지 못했어요.", Toast.LENGTH_SHORT).show();
                    }
                });
    }

    private void deleteAlert() {
        if (alertRuleId == null) {
            return;
        }
        RetrofitClient.getApiService().deleteAlertRule(alertRuleId)
                .enqueue(new Callback<Void>() {
                    @Override
                    public void onResponse(Call<Void> call, Response<Void> response) {
                        alertRuleId = null;
                        alertThreshold = null;
                        updateAlertButton();
                        Toast.makeText(SlotGridActivity.this, "알림을 해제했어요.", Toast.LENGTH_SHORT).show();
                    }

                    @Override
                    public void onFailure(Call<Void> call, Throwable t) {
                        Toast.makeText(SlotGridActivity.this, "네트워크 오류로 해제하지 못했어요.", Toast.LENGTH_SHORT).show();
                    }
                });
    }

    // ===== 내 주차위치 (Phase 2) =====

    private void loadMyLocation() {
        if (!TokenStore.isLoggedIn(this)) {
            return; // 비로그인은 저장 기능 비활성
        }
        RetrofitClient.getApiService().getMyParkingLocation()
                .enqueue(new Callback<ParkingLocationResponse>() {
                    @Override
                    public void onResponse(Call<ParkingLocationResponse> call,
                                           Response<ParkingLocationResponse> response) {
                        if (response.isSuccessful() && response.body() != null) {
                            applyMyLocation(response.body());
                        } else {
                            applyMyLocation(null); // 204 등: 저장된 위치 없음
                        }
                    }

                    @Override
                    public void onFailure(Call<ParkingLocationResponse> call, Throwable t) {
                        // 무시 (조회 실패해도 화면은 동작)
                    }
                });
    }

    private void applyMyLocation(ParkingLocationResponse loc) {
        boolean hereActive = loc != null && loc.active
                && loc.parkingLotId != null && loc.parkingLotId == parkingLotId;
        if (hereActive) {
            slotMap.setMySlot(loc.slotId);
            tvMyLoc.setText("내 주차 위치: " + loc.slotId + "번");
            myLocBar.setVisibility(View.VISIBLE);
        } else {
            slotMap.setMySlot(null);
            myLocBar.setVisibility(View.GONE);
        }
    }

    private void onSlotTapped(SlotItem slot) {
        if (slot.slotId == null) {
            return;
        }
        if (!TokenStore.isLoggedIn(this)) {
            Toast.makeText(this, "내 주차위치 저장은 로그인 후 이용할 수 있어요.", Toast.LENGTH_SHORT).show();
            return;
        }
        new AlertDialog.Builder(this)
                .setMessage(slot.slotId + "번 칸에 내 차를 저장할까요?")
                .setPositiveButton("저장", (d, w) -> saveLocation(slot.slotId))
                .setNegativeButton("취소", null)
                .show();
    }

    private void saveLocation(int slotId) {
        ParkingLocationRequest req =
                new ParkingLocationRequest(parkingLotId, slotId, "내 차", "");
        RetrofitClient.getApiService().saveMyParkingLocation(req)
                .enqueue(new Callback<ParkingLocationResponse>() {
                    @Override
                    public void onResponse(Call<ParkingLocationResponse> call,
                                           Response<ParkingLocationResponse> response) {
                        if (response.isSuccessful() && response.body() != null) {
                            applyMyLocation(response.body());
                            Toast.makeText(SlotGridActivity.this, "내 주차위치를 저장했어요.", Toast.LENGTH_SHORT).show();
                        } else {
                            Toast.makeText(SlotGridActivity.this, "저장 실패 (" + response.code() + ")", Toast.LENGTH_SHORT).show();
                        }
                    }

                    @Override
                    public void onFailure(Call<ParkingLocationResponse> call, Throwable t) {
                        Toast.makeText(SlotGridActivity.this, "네트워크 오류로 저장하지 못했어요.", Toast.LENGTH_SHORT).show();
                    }
                });
    }

    private void releaseLocation() {
        RetrofitClient.getApiService().releaseMyParkingLocation()
                .enqueue(new Callback<ParkingLocationResponse>() {
                    @Override
                    public void onResponse(Call<ParkingLocationResponse> call,
                                           Response<ParkingLocationResponse> response) {
                        slotMap.setMySlot(null);
                        myLocBar.setVisibility(View.GONE);
                        Toast.makeText(SlotGridActivity.this, "주차를 종료했어요.", Toast.LENGTH_SHORT).show();
                    }

                    @Override
                    public void onFailure(Call<ParkingLocationResponse> call, Throwable t) {
                        Toast.makeText(SlotGridActivity.this, "네트워크 오류로 해제하지 못했어요.", Toast.LENGTH_SHORT).show();
                    }
                });
    }

    private void loadImageAsync(String imageUrl, ParkingLotSummary lot) {
        new Thread(() -> {
            Bitmap bmp = null;
            try {
                URL url = new URL(imageUrl);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(6000);
                conn.setReadTimeout(6000);
                conn.connect();
                if (conn.getResponseCode() == 200) {
                    try (InputStream is = conn.getInputStream()) {
                        bmp = BitmapFactory.decodeStream(is);
                    }
                }
                conn.disconnect();
            } catch (Exception ignored) {
                // 사진 로드 실패 시 마커만 유지
            }
            final Bitmap result = bmp;
            runOnUiThread(() -> {
                if (result != null) {
                    slotMap.setData(result, lot.slots);
                }
            });
        }).start();
    }
}

package com.example.pproject;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import com.example.pproject.dto.ParkingStatusResponse;
import com.example.pproject.network.ApiService;
import com.example.pproject.network.RetrofitClient;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class HomeActivity extends AppCompatActivity {

    // UI 컴포넌트 선언
    private TextView tvHomeAvailable, tvHomeOccupied, tvLastUpdate;
    private View btnGoToMap, btnGoToMapBuilder;

    // 5초 자동 갱신을 위한 핸들러
    private Handler handler = new Handler(Looper.getMainLooper());
    private Runnable statusChecker;
    private final int UPDATE_INTERVAL = 5000; // 5초

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);

        // XML ID 연결
        tvHomeAvailable = findViewById(R.id.tvHomeAvailable);
        tvHomeOccupied = findViewById(R.id.tvHomeOccupied);
        tvLastUpdate = findViewById(R.id.tvLastUpdate);
        btnGoToMap = findViewById(R.id.btnGoToMap);
        btnGoToMapBuilder = findViewById(R.id.btnGoToMapBuilder);

        // [주차 현황 확인하기] 버튼 클릭 리스너 -> 지도 화면으로 이동
        btnGoToMap.setOnClickListener(v -> {
            Intent intent = new Intent(HomeActivity.this, ParkingMapActivity.class);
            startActivity(intent);
        });

        btnGoToMapBuilder.setOnClickListener(v -> {
            Intent intent = new Intent(HomeActivity.this, MapBuilderActivity.class);
            startActivity(intent);
        });

        // 주기적 데이터 갱신 로직 정의
        statusChecker = new Runnable() {
            @Override
            public void run() {
                fetchParkingStatus(); // 서버 통신 수행
                handler.postDelayed(this, UPDATE_INTERVAL); // 5초 후 다시 실행
            }
        };

        // 앱이 켜지면 데이터 갱신 시작
        startAutoUpdate();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // 액티비티가 종료될 때 핸들러를 멈춰 메모리 누수 방지
        stopAutoUpdate();
    }

    // 갱신 시작
    private void startAutoUpdate() {
        statusChecker.run();
    }

    // 갱신 정지
    private void stopAutoUpdate() {
        handler.removeCallbacks(statusChecker);
    }

    // 서버로부터 주차 현황 가져오기
    private void fetchParkingStatus() {
        ApiService apiService = RetrofitClient.getApiService();

        apiService.getParkingStatus().enqueue(new Callback<ParkingStatusResponse>() {
            @Override
            public void onResponse(Call<ParkingStatusResponse> call, Response<ParkingStatusResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    // 데이터 수신 성공 시 UI 업데이트
                    updateDashboardUI(response.body());
                } else if (response.code() == 204) {
                    // 데이터 없음 (No Content)
                    tvLastUpdate.setText("데이터 대기 중...");
                } else {
                    // 서버 에러 등
                    android.util.Log.e("HomeActivity", "서버 응답 오류: " + response.code());
                }
            }

            @Override
            public void onFailure(Call<ParkingStatusResponse> call, Throwable t) {
                // 네트워크 연결 실패 시
                tvLastUpdate.setText("네트워크 연결 불안정");
                t.printStackTrace();
            }
        });
    }

    // 받아온 데이터를 바탕으로 텍스트 뷰 갱신
    private void updateDashboardUI(ParkingStatusResponse status) {
        // null 방어 코드 (서버가 일부 구역 정보를 안 보냈을 경우 대비)
        if (status.P1 == null || status.P2 == null || status.P3 == null) return;

        // 1. 전체 잔여 주차 대수 계산 (P1 + P2 + P3)
        int totalAvailable = status.P1.total_available +
                status.P2.total_available +
                status.P3.total_available;

        // 2. 전체 점유(사용 중) 대수 계산
        // 각 구역의 occupied_slots 리스트 크기를 더함
        int occupiedCount = 0;
        if (status.P1.occupied_slots != null) occupiedCount += status.P1.occupied_slots.size();
        if (status.P2.occupied_slots != null) occupiedCount += status.P2.occupied_slots.size();
        if (status.P3.occupied_slots != null) occupiedCount += status.P3.occupied_slots.size();

        // 3. UI 반영
        tvHomeAvailable.setText(String.valueOf(totalAvailable));
        tvHomeOccupied.setText(String.valueOf(occupiedCount));

        // 4. 갱신 시간 표시 (현재 시간 기준)
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.KOREA);
        String currentTime = sdf.format(new Date());
        tvLastUpdate.setText("최근 갱신: " + currentTime);
    }
}
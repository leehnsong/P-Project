package com.example.pproject;

import android.os.Bundle;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.pproject.dto.InAppNotificationResponse;
import com.example.pproject.network.RetrofitClient;

import java.util.List;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

/**
 * 인앱 알림함. GET /api/me/notifications 로 목록을 받아 표시하고,
 * 안 읽은 알림을 탭하면 읽음 처리(PATCH)한다. (폴링 방식, 로그인 필요)
 */
public class NotificationsActivity extends AppCompatActivity {

    private LinearLayout notifContainer;
    private TextView tvNotifEmpty;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_notifications);

        notifContainer = findViewById(R.id.notifContainer);
        tvNotifEmpty = findViewById(R.id.tvNotifEmpty);
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadNotifications();
    }

    private void loadNotifications() {
        if (!TokenStore.isLoggedIn(this)) {
            showEmpty("로그인 후 이용할 수 있어요.");
            return;
        }
        RetrofitClient.getApiService().getNotifications()
                .enqueue(new Callback<List<InAppNotificationResponse>>() {
                    @Override
                    public void onResponse(Call<List<InAppNotificationResponse>> call,
                                           Response<List<InAppNotificationResponse>> response) {
                        if (!response.isSuccessful() || response.body() == null) {
                            showEmpty("알림을 불러오지 못했습니다 (" + response.code() + ")");
                            return;
                        }
                        render(response.body());
                    }

                    @Override
                    public void onFailure(Call<List<InAppNotificationResponse>> call, Throwable t) {
                        showEmpty("네트워크 오류: 백엔드 서버를 확인해 주세요");
                    }
                });
    }

    private void render(List<InAppNotificationResponse> items) {
        notifContainer.removeAllViews();
        if (items.isEmpty()) {
            showEmpty("받은 알림이 없습니다.");
            return;
        }
        tvNotifEmpty.setVisibility(View.GONE);

        for (InAppNotificationResponse n : items) {
            View row = getLayoutInflater().inflate(R.layout.item_notification, notifContainer, false);
            TextView title = row.findViewById(R.id.tvNotifTitle);
            TextView message = row.findViewById(R.id.tvNotifMessage);
            TextView time = row.findViewById(R.id.tvNotifTime);
            View root = row.findViewById(R.id.notifRoot);

            title.setText(n.title != null ? n.title : "알림");
            message.setText(n.message != null ? n.message : "");
            time.setText(formatTime(n.createdAt) + (n.read ? "" : "  • 읽지 않음"));
            root.setBackgroundColor(n.read ? 0x00000000 : 0x1A4F46E5); // 안읽음은 옅은 강조

            if (!n.read && n.id != null) {
                row.setOnClickListener(v -> markRead(n.id));
            }
            notifContainer.addView(row);
        }
    }

    private void markRead(long id) {
        RetrofitClient.getApiService().markNotificationRead(id)
                .enqueue(new Callback<InAppNotificationResponse>() {
                    @Override
                    public void onResponse(Call<InAppNotificationResponse> call,
                                           Response<InAppNotificationResponse> response) {
                        loadNotifications(); // 목록 갱신
                    }

                    @Override
                    public void onFailure(Call<InAppNotificationResponse> call, Throwable t) {
                        // 무시
                    }
                });
    }

    private String formatTime(String iso) {
        if (iso == null) {
            return "";
        }
        // "2026-06-13T20:00:00..." → "2026-06-13 20:00"
        String s = iso.replace('T', ' ');
        return s.length() >= 16 ? s.substring(0, 16) : s;
    }

    private void showEmpty(String message) {
        notifContainer.removeAllViews();
        tvNotifEmpty.setText(message);
        tvNotifEmpty.setVisibility(View.VISIBLE);
    }
}

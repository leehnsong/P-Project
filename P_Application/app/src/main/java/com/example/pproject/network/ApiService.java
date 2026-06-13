package com.example.pproject.network;

import com.example.pproject.dto.LoginRequest;
import com.example.pproject.dto.LoginResponse;
import com.example.pproject.dto.CampusMapResponse;
import com.example.pproject.dto.BuildingDetailResponse;
import com.example.pproject.dto.GeoSearchResult;
import com.example.pproject.dto.ParkingLocationRequest;
import com.example.pproject.dto.ParkingLocationResponse;
import com.example.pproject.dto.ParkingAlertRuleRequest;
import com.example.pproject.dto.ParkingAlertRuleResponse;
import com.example.pproject.dto.InAppNotificationResponse;
import com.example.pproject.dto.UnreadCountResponse;
import com.example.pproject.dto.ParkingStatusResponse;

import java.util.List;
import com.example.pproject.dto.UiConfigResponse;
import com.example.pproject.dto.VoiceAskRequest;
import com.example.pproject.dto.VoiceAskResponse;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.DELETE;
import retrofit2.http.GET;
import retrofit2.http.PATCH;
import retrofit2.http.POST;
import retrofit2.http.Path;
import retrofit2.http.Query;

public interface ApiService {
    // 로그인 (서버가 JSON {token, username} 반환)
    @POST("/auth/login")
    Call<LoginResponse> login(@Body LoginRequest request);

    // 회원가입
    @POST("/auth/register")
    Call<String> register(@Body LoginRequest request);

    // 주차 현황 (서버가 JSON 객체를 반환하므로 DTO 사용)
    @GET("/api/parking/status")
    Call<ParkingStatusResponse> getParkingStatus();

    @GET("/api/ui/config")
    Call<UiConfigResponse> getUiConfig();

    @GET("/api/campus/map")
    Call<CampusMapResponse> getCampusMap();

    // 건물 상세 (주차장 + 슬롯 포함)
    @GET("/api/campus/buildings/{id}")
    Call<BuildingDetailResponse> getBuildingDetail(@Path("id") long buildingId);

    // 장소명 검색 (네이버 지역검색 프록시)
    @GET("/api/geo/search")
    Call<List<GeoSearchResult>> searchPlaces(@Query("query") String query);

    // 음성 질의 (인증 불필요) — {question} 전송 → {answer} 수신
    @POST("/api/voice/ask")
    Call<VoiceAskResponse> askVoice(@Body VoiceAskRequest request);

    // 내 주차위치 (로그인 필요 — 토큰은 인터셉터가 자동 첨부)
    @GET("/api/me/parking-location/current")
    Call<ParkingLocationResponse> getMyParkingLocation();

    @POST("/api/me/parking-location")
    Call<ParkingLocationResponse> saveMyParkingLocation(@Body ParkingLocationRequest request);

    @DELETE("/api/me/parking-location/current")
    Call<ParkingLocationResponse> releaseMyParkingLocation();

    // 빈자리 알림 규칙 (로그인 필요)
    @GET("/api/me/alert-rules")
    Call<List<ParkingAlertRuleResponse>> getAlertRules();

    @POST("/api/me/alert-rules")
    Call<ParkingAlertRuleResponse> createAlertRule(@Body ParkingAlertRuleRequest request);

    @DELETE("/api/me/alert-rules/{ruleId}")
    Call<Void> deleteAlertRule(@Path("ruleId") long ruleId);

    // 인앱 알림함 (로그인 필요)
    @GET("/api/me/notifications")
    Call<List<InAppNotificationResponse>> getNotifications();

    @GET("/api/me/notifications/unread-count")
    Call<UnreadCountResponse> getUnreadCount();

    @PATCH("/api/me/notifications/{id}/read")
    Call<InAppNotificationResponse> markNotificationRead(@Path("id") long notificationId);
}

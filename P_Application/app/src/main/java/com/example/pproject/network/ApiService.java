package com.example.pproject.network;

import com.example.pproject.dto.LoginRequest;
import com.example.pproject.dto.CampusMapResponse;
import com.example.pproject.dto.ParkingStatusResponse;
import com.example.pproject.dto.UiConfigResponse;

import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;

public interface ApiService {
    // 로그인 (서버가 String을 반환하므로 Call<String>)
    @POST("/auth/login")
    Call<String> login(@Body LoginRequest request);

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
}

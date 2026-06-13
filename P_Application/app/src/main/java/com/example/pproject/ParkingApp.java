package com.example.pproject;

import android.app.Application;

import com.example.pproject.network.RetrofitClient;

public class ParkingApp extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        RetrofitClient.init(this); // 토큰 자동 첨부 인터셉터 초기화
    }
}

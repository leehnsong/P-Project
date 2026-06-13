package com.example.pproject.network;

import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.converter.scalars.ScalarsConverterFactory;

public class RetrofitClient {
    // 에뮬레이터 사용 시: 10.0.2.2
    // 실제 폰 사용 시: 실행 중인 PC의 내부 IP (예: 192.168.0.x)
    private static final String BASE_URL = "http://10.0.2.2:8080/";

    private static Retrofit retrofit = null;

    public static ApiService getApiService() {
        if (retrofit == null) {
            retrofit = new Retrofit.Builder()
                    .baseUrl(BASE_URL)
                    // String 반환 처리를 위해 Scalars 먼저 추가
                    .addConverterFactory(ScalarsConverterFactory.create())
                    // JSON 객체 처리를 위해 Gson 추가
                    .addConverterFactory(GsonConverterFactory.create())
                    .build();
        }
        return retrofit.create(ApiService.class);
    }
}

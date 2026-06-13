package com.example.pproject.network;

import android.content.Context;

import com.example.pproject.TokenStore;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import retrofit2.converter.scalars.ScalarsConverterFactory;

public class RetrofitClient {
    // 에뮬레이터 사용 시: 10.0.2.2
    // 실제 폰 사용 시: 실행 중인 PC의 내부 IP (예: 192.168.0.x)
    private static final String BASE_URL = "http://10.0.2.2:8080/";

    private static Context appContext;
    private static Retrofit retrofit = null;

    /** Application에서 1회 호출. 토큰 자동 첨부 인터셉터가 이 컨텍스트로 토큰을 읽는다. */
    public static void init(Context context) {
        appContext = context.getApplicationContext();
    }

    /** 이미지 등 절대 URL 조립용. 끝의 '/'는 제거해서 반환. */
    public static String getServerOrigin() {
        return BASE_URL.endsWith("/") ? BASE_URL.substring(0, BASE_URL.length() - 1) : BASE_URL;
    }

    public static ApiService getApiService() {
        if (retrofit == null) {
            OkHttpClient client = new OkHttpClient.Builder()
                    .addInterceptor(chain -> {
                        Request request = chain.request();
                        String token = appContext != null ? TokenStore.getToken(appContext) : null;
                        if (token != null && !token.isEmpty()) {
                            request = request.newBuilder()
                                    .header("Authorization", "Bearer " + token)
                                    .build();
                        }
                        return chain.proceed(request);
                    })
                    .build();

            retrofit = new Retrofit.Builder()
                    .baseUrl(BASE_URL)
                    .client(client)
                    // String 반환 처리를 위해 Scalars 먼저 추가
                    .addConverterFactory(ScalarsConverterFactory.create())
                    // JSON 객체 처리를 위해 Gson 추가
                    .addConverterFactory(GsonConverterFactory.create())
                    .build();
        }
        return retrofit.create(ApiService.class);
    }
}

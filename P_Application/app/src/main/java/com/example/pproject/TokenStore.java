package com.example.pproject;

import android.content.Context;
import android.content.SharedPreferences;

/**
 * 로그인 시 MainActivity가 저장한 JWT 토큰/사용자명을 읽고 지우는 헬퍼.
 * 저장 위치: SharedPreferences("AuthData") - "jwt_token", "username"
 */
public class TokenStore {

    private static final String PREF = "AuthData";
    private static final String KEY_TOKEN = "jwt_token";
    private static final String KEY_USERNAME = "username";

    private static SharedPreferences prefs(Context ctx) {
        return ctx.getSharedPreferences(PREF, Context.MODE_PRIVATE);
    }

    public static String getToken(Context ctx) {
        return prefs(ctx).getString(KEY_TOKEN, null);
    }

    public static String getUsername(Context ctx) {
        return prefs(ctx).getString(KEY_USERNAME, null);
    }

    public static boolean isLoggedIn(Context ctx) {
        String t = getToken(ctx);
        return t != null && !t.isEmpty();
    }

    public static void clear(Context ctx) {
        prefs(ctx).edit().remove(KEY_TOKEN).remove(KEY_USERNAME).apply();
    }
}

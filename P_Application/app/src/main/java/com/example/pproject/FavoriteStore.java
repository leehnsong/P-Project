package com.example.pproject;

import android.content.Context;
import android.content.SharedPreferences;

import java.util.HashSet;
import java.util.Set;

/**
 * 즐겨찾기한 주차장(parkingLotId)을 기기에 로컬 저장.
 * 백엔드에 즐겨찾기 API가 없으므로 SharedPreferences로 관리한다.
 */
public class FavoriteStore {

    private static final String PREF = "favorites";
    private static final String KEY = "lot_ids";

    private static SharedPreferences prefs(Context ctx) {
        return ctx.getSharedPreferences(PREF, Context.MODE_PRIVATE);
    }

    public static Set<String> getIds(Context ctx) {
        // SharedPreferences가 돌려준 Set은 수정 금지 → 복사본 반환
        return new HashSet<>(prefs(ctx).getStringSet(KEY, new HashSet<>()));
    }

    public static boolean isFavorite(Context ctx, long lotId) {
        return getIds(ctx).contains(String.valueOf(lotId));
    }

    public static void toggle(Context ctx, long lotId) {
        Set<String> ids = getIds(ctx);
        String key = String.valueOf(lotId);
        if (ids.contains(key)) {
            ids.remove(key);
        } else {
            ids.add(key);
        }
        prefs(ctx).edit().putStringSet(KEY, ids).apply();
    }
}

package com.example.pproject;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import com.example.pproject.dto.SlotItem;

import java.util.List;

/**
 * 주차장 원본 사진 위에 슬롯을 실제 위치(854x480 좌표계)대로 색상 마커로 오버레이한다.
 * 웹과 동일한 기준: center[x]/854, center[y]/480 비율로 배치.
 */
public class SlotMapView extends View {

    private static final float REF_W = 854f;
    private static final float REF_H = 480f;

    private Bitmap bgBitmap; // 흐리게 처리된 배경(다운스케일본)
    private List<SlotItem> slots;
    private Integer mySlotId; // 내 주차위치로 저장된 슬롯(있으면 강조)

    public interface OnSlotClickListener {
        void onSlotClick(SlotItem slot);
    }

    private OnSlotClickListener slotClickListener;
    private final Paint myRingPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final Paint fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint borderPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint bgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Rect srcRect = new Rect();
    private final RectF dstRect = new RectF();

    public SlotMapView(Context context) {
        super(context);
        init();
    }

    public SlotMapView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setColor(0xFFFFFFFF);
        borderPaint.setStrokeWidth(dp(1.5f));
        textPaint.setColor(0xFFFFFFFF);
        textPaint.setTextAlign(Paint.Align.CENTER);
        textPaint.setFakeBoldText(true);
        textPaint.setTextSize(dp(11));
        bgPaint.setFilterBitmap(true); // 업스케일 시 부드럽게(블러 효과)
        myRingPaint.setStyle(Paint.Style.STROKE);
        myRingPaint.setColor(0xFFFFC107); // 내 위치 강조 링(노랑)
        myRingPaint.setStrokeWidth(dp(3.5f));
        setClickable(true);
    }

    public void setData(Bitmap bitmap, List<SlotItem> slots) {
        this.bgBitmap = bitmap != null ? makeBlurred(bitmap) : null;
        this.slots = slots;
        requestLayout();
        invalidate();
    }

    public void setMySlot(Integer slotId) {
        this.mySlotId = slotId;
        invalidate();
    }

    public void setOnSlotClickListener(OnSlotClickListener listener) {
        this.slotClickListener = listener;
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (event.getAction() == MotionEvent.ACTION_UP && slots != null && slotClickListener != null) {
            float tx = event.getX();
            float ty = event.getY();
            int w = getWidth();
            int h = getHeight();
            float hitR = dp(20);
            SlotItem nearest = null;
            float best = Float.MAX_VALUE;
            for (SlotItem slot : slots) {
                if (slot.center == null || slot.center.size() < 2) continue;
                float cx = (1f - (float) (slot.center.get(1) / REF_H)) * w;
                float cy = (float) (slot.center.get(0) / REF_W) * h;
                float d = (float) Math.hypot(tx - cx, ty - cy);
                if (d < best) { best = d; nearest = slot; }
            }
            if (nearest != null && best <= hitR) {
                slotClickListener.onSlotClick(nearest);
                performClick();
                return true;
            }
        }
        return super.onTouchEvent(event);
    }

    @Override
    public boolean performClick() {
        return super.performClick();
    }

    /** 작게 줄였다가 크게 그려 흐릿하게(차량이 또렷이 보이지 않게) 만든다. */
    private Bitmap makeBlurred(Bitmap src) {
        int targetW = 110;
        int targetH = Math.max(1, Math.round(targetW * (float) src.getHeight() / src.getWidth()));
        try {
            return Bitmap.createScaledBitmap(src, targetW, targetH, true);
        } catch (Exception e) {
            return src;
        }
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        // 가로(854x480) 사진을 시계방향 90도로 세워서 세로 화면을 꽉 채움 → 비율 480:854
        int width = MeasureSpec.getSize(widthMeasureSpec);
        int height = Math.round(width * REF_W / REF_H);
        setMeasuredDimension(width, height);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int w = getWidth();
        int h = getHeight();

        // 흐리게 처리된 사진을 시계방향 90도 회전해 뷰 전체를 채움
        if (bgBitmap != null) {
            canvas.save();
            canvas.translate(w, 0);
            canvas.rotate(90);
            srcRect.set(0, 0, bgBitmap.getWidth(), bgBitmap.getHeight());
            dstRect.set(0, 0, h, w); // 회전 좌표계 기준 (a축→화면 세로, b축→화면 가로)
            canvas.drawBitmap(bgBitmap, srcRect, dstRect, bgPaint);
            canvas.restore();
            // 아주 옅은 스크림만 덮어 마커가 도드라지게
            canvas.drawColor(0x22FFFFFF);
        } else {
            canvas.drawColor(0xFFEDEDED);
        }

        if (slots == null) {
            return;
        }

        // 마커는 회전된 화면 좌표로 직접 계산(숫자는 똑바로 유지)
        float r = dp(13);
        for (SlotItem slot : slots) {
            if (slot.center == null || slot.center.size() < 2) {
                continue;
            }
            float nx = (float) (slot.center.get(0) / REF_W); // 0~1 (원본 가로)
            float ny = (float) (slot.center.get(1) / REF_H); // 0~1 (원본 세로)
            float cx = (1f - ny) * w; // 시계방향 90도 변환
            float cy = nx * h;

            fillPaint.setColor(colorFor(slot));
            canvas.drawCircle(cx, cy, r, fillPaint);
            canvas.drawCircle(cx, cy, r, borderPaint);

            if (mySlotId != null && slot.slotId != null && slot.slotId.equals(mySlotId)) {
                canvas.drawCircle(cx, cy, r + dp(3.5f), myRingPaint);
            }

            String label = slot.slotId != null ? String.valueOf(slot.slotId) : "";
            float ty = cy - (textPaint.descent() + textPaint.ascent()) / 2f;
            canvas.drawText(label, cx, ty, textPaint);
        }
    }

    private int colorFor(SlotItem slot) {
        boolean occupied = "occupied".equalsIgnoreCase(slot.status);
        boolean disabled = "disabled".equalsIgnoreCase(slot.type);
        if (occupied) {
            return 0xE6F44336; // 사용중 - 빨강
        }
        if (disabled) {
            return 0xE63B82F6; // 장애인석 빈자리 - 파랑
        }
        return 0xE64CAF50; // 비어있음 - 초록
    }

    private float dp(float value) {
        return value * getResources().getDisplayMetrics().density;
    }
}

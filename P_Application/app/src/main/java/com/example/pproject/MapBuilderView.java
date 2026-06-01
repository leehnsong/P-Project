package com.example.pproject;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import androidx.annotation.DrawableRes;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Touch-based parking map builder inspired by the OpenCV labeling scripts.
 * It stores rectangles in original bitmap coordinates so the exported JSON can
 * be reused by the Python mapping tools or by a server-side map pipeline.
 */
public class MapBuilderView extends View {

    public interface OnSlotChangedListener {
        void onSlotChanged(int slotCount, String currentZoneLabel);
    }

    private static final float MIN_SLOT_SIZE_PX = 8f;

    private final Paint bitmapPaint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final Paint slotPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint disabledSlotPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint previewPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint hintPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final Map<String, List<ParkingSlotShape>> slotsByZone = new HashMap<>();
    private final Map<String, Integer> nextSlotByZone = new HashMap<>();

    private Bitmap mapBitmap;
    private RectF bitmapDest = new RectF();
    private String currentZoneKey = "partition1";
    private String currentZoneLabel = "P1";
    private boolean nextSlotDisabled;
    private float startBitmapX;
    private float startBitmapY;
    private RectF previewBitmapRect;
    private OnSlotChangedListener listener;

    public MapBuilderView(Context context) {
        super(context);
        init();
    }

    public MapBuilderView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public MapBuilderView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        setBackgroundColor(Color.WHITE);

        slotPaint.setStyle(Paint.Style.STROKE);
        slotPaint.setStrokeWidth(4f);
        slotPaint.setColor(Color.rgb(76, 175, 80));

        disabledSlotPaint.setStyle(Paint.Style.STROKE);
        disabledSlotPaint.setStrokeWidth(4f);
        disabledSlotPaint.setColor(Color.rgb(255, 193, 7));

        previewPaint.setStyle(Paint.Style.STROKE);
        previewPaint.setStrokeWidth(5f);
        previewPaint.setColor(Color.rgb(79, 70, 229));

        textPaint.setColor(Color.WHITE);
        textPaint.setTextAlign(Paint.Align.CENTER);
        textPaint.setTextSize(28f);
        textPaint.setFakeBoldText(true);

        hintPaint.setColor(Color.rgb(97, 97, 97));
        hintPaint.setTextAlign(Paint.Align.CENTER);
        hintPaint.setTextSize(34f);

        slotsByZone.put("partition1", new ArrayList<>());
        slotsByZone.put("partition2", new ArrayList<>());
        slotsByZone.put("partition3", new ArrayList<>());
        nextSlotByZone.put("partition1", 1);
        nextSlotByZone.put("partition2", 42);
        nextSlotByZone.put("partition3", 73);
    }

    public void setOnSlotChangedListener(OnSlotChangedListener listener) {
        this.listener = listener;
        notifySlotChanged();
    }

    public void setZone(String zoneKey, String zoneLabel, @DrawableRes int drawableRes) {
        currentZoneKey = zoneKey;
        currentZoneLabel = zoneLabel;
        mapBitmap = BitmapFactory.decodeResource(getResources(), drawableRes);
        previewBitmapRect = null;
        notifySlotChanged();
        invalidate();
    }

    public void setNextSlotDisabled(boolean nextSlotDisabled) {
        this.nextSlotDisabled = nextSlotDisabled;
    }

    public void undoLastSlot() {
        List<ParkingSlotShape> slots = getCurrentSlots();
        if (slots.isEmpty()) {
            return;
        }
        slots.remove(slots.size() - 1);
        rebuildNextSlotForCurrentZone();
        notifySlotChanged();
        invalidate();
    }

    public void clearCurrentZone() {
        getCurrentSlots().clear();
        resetCurrentZoneStartSlot();
        notifySlotChanged();
        invalidate();
    }

    public String exportMappingJson() throws JSONException {
        JSONObject root = new JSONObject();
        root.put("partition1", toJsonArray(slotsByZone.get("partition1")));
        root.put("partition2", toJsonArray(slotsByZone.get("partition2")));
        root.put("partition3", toJsonArray(slotsByZone.get("partition3")));
        return root.toString(2);
    }

    private JSONArray toJsonArray(List<ParkingSlotShape> slots) throws JSONException {
        JSONArray array = new JSONArray();
        if (slots == null) {
            return array;
        }
        for (ParkingSlotShape slot : slots) {
            JSONObject item = new JSONObject();
            item.put("slot", slot.slotNumber);
            item.put("points", slot.toPointsJson());
            item.put("disabled", slot.disabled);
            array.put(item);
        }
        return array;
    }

    private List<ParkingSlotShape> getCurrentSlots() {
        List<ParkingSlotShape> slots = slotsByZone.get(currentZoneKey);
        if (slots == null) {
            slots = new ArrayList<>();
            slotsByZone.put(currentZoneKey, slots);
        }
        return slots;
    }

    private void rebuildNextSlotForCurrentZone() {
        List<ParkingSlotShape> slots = getCurrentSlots();
        if (slots.isEmpty()) {
            resetCurrentZoneStartSlot();
            return;
        }
        int max = slots.get(0).slotNumber;
        for (ParkingSlotShape slot : slots) {
            max = Math.max(max, slot.slotNumber);
        }
        nextSlotByZone.put(currentZoneKey, max + 1);
    }

    private void resetCurrentZoneStartSlot() {
        if ("partition1".equals(currentZoneKey)) {
            nextSlotByZone.put(currentZoneKey, 1);
        } else if ("partition2".equals(currentZoneKey)) {
            nextSlotByZone.put(currentZoneKey, 42);
        } else {
            nextSlotByZone.put(currentZoneKey, 73);
        }
    }

    private void notifySlotChanged() {
        if (listener != null) {
            listener.onSlotChanged(getCurrentSlots().size(), currentZoneLabel);
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (mapBitmap == null) {
            canvas.drawText("구역을 선택하세요", getWidth() / 2f, getHeight() / 2f, hintPaint);
            return;
        }

        updateBitmapDest();
        canvas.drawBitmap(mapBitmap, null, bitmapDest, bitmapPaint);

        for (ParkingSlotShape slot : getCurrentSlots()) {
            drawSlot(canvas, slot, slot.disabled ? disabledSlotPaint : slotPaint);
        }

        if (previewBitmapRect != null) {
            RectF previewViewRect = bitmapToView(previewBitmapRect);
            canvas.drawRect(previewViewRect, previewPaint);
        }
    }

    private void drawSlot(Canvas canvas, ParkingSlotShape slot, Paint paint) {
        RectF viewRect = bitmapToView(slot.rect);
        canvas.drawRect(viewRect, paint);

        Paint fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        fillPaint.setColor(slot.disabled ? Color.rgb(255, 193, 7) : Color.rgb(76, 175, 80));
        fillPaint.setAlpha(210);
        canvas.drawCircle(viewRect.centerX(), viewRect.centerY(), 24f, fillPaint);
        canvas.drawText(String.valueOf(slot.slotNumber), viewRect.centerX(), viewRect.centerY() + 10f, textPaint);
    }

    private void updateBitmapDest() {
        float viewWidth = getWidth() - getPaddingLeft() - getPaddingRight();
        float viewHeight = getHeight() - getPaddingTop() - getPaddingBottom();
        float bitmapWidth = mapBitmap.getWidth();
        float bitmapHeight = mapBitmap.getHeight();
        float scale = Math.min(viewWidth / bitmapWidth, viewHeight / bitmapHeight);
        float scaledWidth = bitmapWidth * scale;
        float scaledHeight = bitmapHeight * scale;
        float left = getPaddingLeft() + (viewWidth - scaledWidth) / 2f;
        float top = getPaddingTop() + (viewHeight - scaledHeight) / 2f;
        bitmapDest.set(left, top, left + scaledWidth, top + scaledHeight);
    }

    private RectF bitmapToView(RectF bitmapRect) {
        float scaleX = bitmapDest.width() / mapBitmap.getWidth();
        float scaleY = bitmapDest.height() / mapBitmap.getHeight();
        return new RectF(
                bitmapDest.left + bitmapRect.left * scaleX,
                bitmapDest.top + bitmapRect.top * scaleY,
                bitmapDest.left + bitmapRect.right * scaleX,
                bitmapDest.top + bitmapRect.bottom * scaleY
        );
    }

    private float viewToBitmapX(float viewX) {
        return (viewX - bitmapDest.left) * mapBitmap.getWidth() / bitmapDest.width();
    }

    private float viewToBitmapY(float viewY) {
        return (viewY - bitmapDest.top) * mapBitmap.getHeight() / bitmapDest.height();
    }

    private boolean isInsideBitmap(float x, float y) {
        return bitmapDest.contains(x, y);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (mapBitmap == null) {
            return false;
        }
        updateBitmapDest();
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                if (!isInsideBitmap(x, y)) {
                    return false;
                }
                startBitmapX = clamp(viewToBitmapX(x), 0f, mapBitmap.getWidth());
                startBitmapY = clamp(viewToBitmapY(y), 0f, mapBitmap.getHeight());
                previewBitmapRect = new RectF(startBitmapX, startBitmapY, startBitmapX, startBitmapY);
                invalidate();
                return true;
            case MotionEvent.ACTION_MOVE:
                if (previewBitmapRect != null) {
                    updatePreviewRect(x, y);
                    invalidate();
                }
                return true;
            case MotionEvent.ACTION_UP:
                if (previewBitmapRect != null) {
                    updatePreviewRect(x, y);
                    addPreviewSlotIfValid();
                    previewBitmapRect = null;
                    invalidate();
                }
                return true;
            case MotionEvent.ACTION_CANCEL:
                previewBitmapRect = null;
                invalidate();
                return true;
            default:
                return super.onTouchEvent(event);
        }
    }

    private void updatePreviewRect(float viewX, float viewY) {
        float endBitmapX = clamp(viewToBitmapX(viewX), 0f, mapBitmap.getWidth());
        float endBitmapY = clamp(viewToBitmapY(viewY), 0f, mapBitmap.getHeight());
        previewBitmapRect.set(
                Math.min(startBitmapX, endBitmapX),
                Math.min(startBitmapY, endBitmapY),
                Math.max(startBitmapX, endBitmapX),
                Math.max(startBitmapY, endBitmapY)
        );
    }

    private void addPreviewSlotIfValid() {
        if (previewBitmapRect.width() < MIN_SLOT_SIZE_PX || previewBitmapRect.height() < MIN_SLOT_SIZE_PX) {
            return;
        }
        int nextSlot = nextSlotByZone.containsKey(currentZoneKey) ? nextSlotByZone.get(currentZoneKey) : 1;
        ParkingSlotShape shape = new ParkingSlotShape(nextSlot, new RectF(previewBitmapRect), nextSlotDisabled);
        getCurrentSlots().add(shape);
        nextSlotByZone.put(currentZoneKey, nextSlot + 1);
        notifySlotChanged();
    }

    private float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    public String getCurrentNextSlotLabel() {
        Integer nextSlot = nextSlotByZone.get(currentZoneKey);
        return String.format(Locale.KOREA, "%s 다음 슬롯: %d", currentZoneLabel, nextSlot == null ? 1 : nextSlot);
    }

    private static class ParkingSlotShape {
        private final int slotNumber;
        private final RectF rect;
        private final boolean disabled;

        private ParkingSlotShape(int slotNumber, RectF rect, boolean disabled) {
            this.slotNumber = slotNumber;
            this.rect = rect;
            this.disabled = disabled;
        }

        private JSONArray toPointsJson() throws JSONException {
            JSONArray points = new JSONArray();
            points.put(point(rect.left, rect.top));
            points.put(point(rect.right, rect.top));
            points.put(point(rect.right, rect.bottom));
            points.put(point(rect.left, rect.bottom));
            return points;
        }

        private JSONArray point(float x, float y) throws JSONException {
            JSONArray point = new JSONArray();
            point.put(Math.round(x));
            point.put(Math.round(y));
            return point;
        }
    }
}

package com.example.pproject;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
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

public class MapBuilderView extends View {

    public interface OnSlotChangedListener {
        void onSlotChanged(int slotCount, String zoneLabel, int selectedSlotNumber);
    }

    private static final float DEFAULT_SLOT_WIDTH = 40f;
    private static final float DEFAULT_SLOT_HEIGHT = 70f;
    private static final float MIN_SLOT_SIZE = 5f;
    private static final float MOVE_THRESHOLD = 2f;

    private final Paint bitmapPaint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
    private final Paint normalPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint disabledPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint selectedPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint hintPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final RectF bitmapDest = new RectF();
    private final Map<String, List<ParkingSlot>> slotsByZone = new HashMap<>();
    private final Map<String, Integer> nextSlotByZone = new HashMap<>();

    private Bitmap mapBitmap;
    private String currentZoneKey = "partition1";
    private String currentZoneLabel = "P1";
    private int selectedIndex = -1;
    private boolean nextSlotDisabled;
    private boolean dragging;
    private boolean movedDuringGesture;
    private boolean createdDuringGesture;
    private float gestureStartX;
    private float gestureStartY;
    private float currentWidth = DEFAULT_SLOT_WIDTH;
    private float currentHeight = DEFAULT_SLOT_HEIGHT;
    private int currentAngle;
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

        normalPaint.setStyle(Paint.Style.STROKE);
        normalPaint.setStrokeWidth(3f);
        normalPaint.setColor(Color.rgb(213, 0, 0));

        disabledPaint.setStyle(Paint.Style.STROKE);
        disabledPaint.setStrokeWidth(3f);
        disabledPaint.setColor(Color.rgb(25, 118, 210));

        selectedPaint.setStyle(Paint.Style.STROKE);
        selectedPaint.setStrokeWidth(5f);
        selectedPaint.setColor(Color.rgb(0, 180, 90));

        textPaint.setColor(Color.WHITE);
        textPaint.setTextAlign(Paint.Align.CENTER);
        textPaint.setTextSize(24f);
        textPaint.setFakeBoldText(true);

        hintPaint.setColor(Color.rgb(80, 80, 80));
        hintPaint.setTextAlign(Paint.Align.CENTER);
        hintPaint.setTextSize(30f);

        slotsByZone.put("partition1", new ArrayList<>());
        slotsByZone.put("partition2", new ArrayList<>());
        slotsByZone.put("partition3", new ArrayList<>());
        nextSlotByZone.put("partition1", 1);
        nextSlotByZone.put("partition2", 42);
        nextSlotByZone.put("partition3", 67);
    }

    public void setOnSlotChangedListener(OnSlotChangedListener listener) {
        this.listener = listener;
        notifySlotChanged();
    }

    public void setZone(String zoneKey, String zoneLabel, @DrawableRes int drawableRes) {
        currentZoneKey = zoneKey;
        currentZoneLabel = zoneLabel;
        mapBitmap = BitmapFactory.decodeResource(getResources(), drawableRes);
        selectedIndex = getCurrentSlots().isEmpty() ? -1 : 0;
        syncControlsFromSelected();
        notifySlotChanged();
        invalidate();
    }

    public void setSelectedSlotDisabled(boolean disabled) {
        ParkingSlot selected = getSelectedSlot();
        if (selected == null) {
            nextSlotDisabled = disabled;
            return;
        }
        selected.disabled = disabled;
        notifySlotChanged();
        invalidate();
    }

    public void setSelectedWidth(int width) {
        currentWidth = Math.max(MIN_SLOT_SIZE, width);
        ParkingSlot selected = getSelectedSlot();
        if (selected != null) {
            selected.w = currentWidth;
            invalidate();
        }
    }

    public void setSelectedHeight(int height) {
        currentHeight = Math.max(MIN_SLOT_SIZE, height);
        ParkingSlot selected = getSelectedSlot();
        if (selected != null) {
            selected.h = currentHeight;
            invalidate();
        }
    }

    public void setSelectedAngle(int angle) {
        currentAngle = ((angle % 360) + 360) % 360;
        ParkingSlot selected = getSelectedSlot();
        if (selected != null) {
            selected.angle = currentAngle;
            invalidate();
        }
    }

    public int getSelectedWidth() {
        return Math.round(currentWidth);
    }

    public int getSelectedHeight() {
        return Math.round(currentHeight);
    }

    public int getSelectedAngle() {
        return currentAngle;
    }

    public boolean isSelectedSlotDisabled() {
        ParkingSlot selected = getSelectedSlot();
        return selected != null ? selected.disabled : nextSlotDisabled;
    }

    public void deleteSelectedSlot() {
        List<ParkingSlot> slots = getCurrentSlots();
        if (selectedIndex < 0 || selectedIndex >= slots.size()) {
            return;
        }
        slots.remove(selectedIndex);
        if (slots.isEmpty()) {
            selectedIndex = -1;
        } else if (selectedIndex >= slots.size()) {
            selectedIndex = slots.size() - 1;
        }
        rebuildNextSlotForCurrentZone();
        syncControlsFromSelected();
        notifySlotChanged();
        invalidate();
    }

    public void undoLastSlot() {
        List<ParkingSlot> slots = getCurrentSlots();
        if (slots.isEmpty()) {
            return;
        }
        slots.remove(slots.size() - 1);
        if (selectedIndex >= slots.size()) {
            selectedIndex = slots.size() - 1;
        }
        rebuildNextSlotForCurrentZone();
        syncControlsFromSelected();
        notifySlotChanged();
        invalidate();
    }

    public void clearCurrentZone() {
        getCurrentSlots().clear();
        selectedIndex = -1;
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

    public String getCurrentNextSlotLabel() {
        Integer nextSlot = nextSlotByZone.get(currentZoneKey);
        int selectedNumber = getSelectedSlotNumber();
        String selectedText = selectedNumber > 0 ? String.format(Locale.KOREA, " / 선택: %d", selectedNumber) : "";
        return String.format(Locale.KOREA, "%s 다음 슬롯: %d%s", currentZoneLabel, nextSlot == null ? 1 : nextSlot, selectedText);
    }

    private JSONArray toJsonArray(List<ParkingSlot> slots) throws JSONException {
        JSONArray array = new JSONArray();
        if (slots == null) {
            return array;
        }
        for (ParkingSlot slot : slots) {
            JSONObject item = new JSONObject();
            JSONArray center = new JSONArray();
            center.put((double) slot.cx);
            center.put((double) slot.cy);

            item.put("slot", slot.slotNumber);
            item.put("type", slot.disabled ? "disabled" : "normal");
            item.put("center", center);
            item.put("w", Math.round(slot.w));
            item.put("h", Math.round(slot.h));
            item.put("angle", slot.angle);
            array.put(item);
        }
        return array;
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

        List<ParkingSlot> slots = getCurrentSlots();
        for (int i = 0; i < slots.size(); i++) {
            drawSlot(canvas, slots.get(i), i == selectedIndex);
        }
    }

    private void drawSlot(Canvas canvas, ParkingSlot slot, boolean selected) {
        PointF[] points = getViewPoints(slot);
        Path path = new Path();
        path.moveTo(points[0].x, points[0].y);
        for (int i = 1; i < points.length; i++) {
            path.lineTo(points[i].x, points[i].y);
        }
        path.close();

        fillPaint.setStyle(Paint.Style.FILL);
        fillPaint.setColor(slot.disabled ? Color.rgb(25, 118, 210) : Color.rgb(213, 0, 0));
        fillPaint.setAlpha(selected ? 120 : 70);
        canvas.drawPath(path, fillPaint);
        canvas.drawPath(path, slot.disabled ? disabledPaint : normalPaint);
        if (selected) {
            canvas.drawPath(path, selectedPaint);
        }

        PointF center = bitmapToView(slot.cx, slot.cy);
        canvas.drawText(String.valueOf(slot.slotNumber), center.x, center.y + 8f, textPaint);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (mapBitmap == null) {
            return false;
        }
        updateBitmapDest();
        if (!isInsideBitmap(event.getX(), event.getY()) && event.getAction() == MotionEvent.ACTION_DOWN) {
            return false;
        }

        PointF bitmapPoint = viewToBitmap(event.getX(), event.getY());
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                gestureStartX = bitmapPoint.x;
                gestureStartY = bitmapPoint.y;
                movedDuringGesture = false;
                createdDuringGesture = false;
                int hitIndex = findSlotAt(bitmapPoint.x, bitmapPoint.y);
                if (hitIndex >= 0) {
                    selectedIndex = hitIndex;
                } else {
                    addSlot(bitmapPoint.x, bitmapPoint.y);
                    createdDuringGesture = true;
                }
                dragging = true;
                syncControlsFromSelected();
                notifySlotChanged();
                invalidate();
                return true;
            case MotionEvent.ACTION_MOVE:
                ParkingSlot selected = getSelectedSlot();
                if (dragging && selected != null) {
                    if (Math.abs(bitmapPoint.x - gestureStartX) > MOVE_THRESHOLD
                            || Math.abs(bitmapPoint.y - gestureStartY) > MOVE_THRESHOLD) {
                        movedDuringGesture = true;
                    }
                    selected.cx = clamp(bitmapPoint.x, 0f, mapBitmap.getWidth());
                    selected.cy = clamp(bitmapPoint.y, 0f, mapBitmap.getHeight());
                    invalidate();
                }
                return true;
            case MotionEvent.ACTION_UP:
                if (dragging && !movedDuringGesture && !createdDuringGesture) {
                    ParkingSlot tapped = getSelectedSlot();
                    if (tapped != null) {
                        tapped.disabled = !tapped.disabled;
                        nextSlotDisabled = tapped.disabled;
                    }
                }
                dragging = false;
                movedDuringGesture = false;
                createdDuringGesture = false;
                syncControlsFromSelected();
                notifySlotChanged();
                invalidate();
                return true;
            case MotionEvent.ACTION_CANCEL:
                dragging = false;
                movedDuringGesture = false;
                createdDuringGesture = false;
                return true;
            default:
                return super.onTouchEvent(event);
        }
    }

    private void addSlot(float cx, float cy) {
        int nextSlot = nextSlotByZone.containsKey(currentZoneKey) ? nextSlotByZone.get(currentZoneKey) : 1;
        ParkingSlot slot = new ParkingSlot(nextSlot, cx, cy, currentWidth, currentHeight, currentAngle, nextSlotDisabled);
        List<ParkingSlot> slots = getCurrentSlots();
        slots.add(slot);
        selectedIndex = slots.size() - 1;
        nextSlotByZone.put(currentZoneKey, nextSlot + 1);
    }

    private int findSlotAt(float x, float y) {
        List<ParkingSlot> slots = getCurrentSlots();
        for (int i = slots.size() - 1; i >= 0; i--) {
            if (isInsideSlot(slots.get(i), x, y)) {
                return i;
            }
        }
        return -1;
    }

    private boolean isInsideSlot(ParkingSlot slot, float x, float y) {
        double angle = Math.toRadians(-slot.angle);
        double dx = x - slot.cx;
        double dy = y - slot.cy;
        double localX = dx * Math.cos(angle) - dy * Math.sin(angle);
        double localY = dx * Math.sin(angle) + dy * Math.cos(angle);
        return Math.abs(localX) <= slot.w / 2f && Math.abs(localY) <= slot.h / 2f;
    }

    private PointF[] getViewPoints(ParkingSlot slot) {
        PointF[] points = new PointF[4];
        double angle = Math.toRadians(slot.angle);
        double cos = Math.cos(angle);
        double sin = Math.sin(angle);
        float halfW = slot.w / 2f;
        float halfH = slot.h / 2f;
        float[][] corners = {
                {-halfW, -halfH},
                {halfW, -halfH},
                {halfW, halfH},
                {-halfW, halfH}
        };

        for (int i = 0; i < corners.length; i++) {
            float rx = (float) (corners[i][0] * cos - corners[i][1] * sin);
            float ry = (float) (corners[i][0] * sin + corners[i][1] * cos);
            points[i] = bitmapToView(slot.cx + rx, slot.cy + ry);
        }
        return points;
    }

    private PointF bitmapToView(float bitmapX, float bitmapY) {
        float scaleX = bitmapDest.width() / mapBitmap.getWidth();
        float scaleY = bitmapDest.height() / mapBitmap.getHeight();
        return new PointF(bitmapDest.left + bitmapX * scaleX, bitmapDest.top + bitmapY * scaleY);
    }

    private PointF viewToBitmap(float viewX, float viewY) {
        float bitmapX = (viewX - bitmapDest.left) * mapBitmap.getWidth() / bitmapDest.width();
        float bitmapY = (viewY - bitmapDest.top) * mapBitmap.getHeight() / bitmapDest.height();
        return new PointF(clamp(bitmapX, 0f, mapBitmap.getWidth()), clamp(bitmapY, 0f, mapBitmap.getHeight()));
    }

    private boolean isInsideBitmap(float x, float y) {
        return bitmapDest.contains(x, y);
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

    private List<ParkingSlot> getCurrentSlots() {
        List<ParkingSlot> slots = slotsByZone.get(currentZoneKey);
        if (slots == null) {
            slots = new ArrayList<>();
            slotsByZone.put(currentZoneKey, slots);
        }
        return slots;
    }

    private ParkingSlot getSelectedSlot() {
        List<ParkingSlot> slots = getCurrentSlots();
        if (selectedIndex < 0 || selectedIndex >= slots.size()) {
            return null;
        }
        return slots.get(selectedIndex);
    }

    private int getSelectedSlotNumber() {
        ParkingSlot selected = getSelectedSlot();
        return selected == null ? -1 : selected.slotNumber;
    }

    private void syncControlsFromSelected() {
        ParkingSlot selected = getSelectedSlot();
        if (selected == null) {
            return;
        }
        currentWidth = selected.w;
        currentHeight = selected.h;
        currentAngle = selected.angle;
        nextSlotDisabled = selected.disabled;
    }

    private void rebuildNextSlotForCurrentZone() {
        List<ParkingSlot> slots = getCurrentSlots();
        if (slots.isEmpty()) {
            resetCurrentZoneStartSlot();
            return;
        }
        int maxSlot = slots.get(0).slotNumber;
        for (ParkingSlot slot : slots) {
            maxSlot = Math.max(maxSlot, slot.slotNumber);
        }
        nextSlotByZone.put(currentZoneKey, maxSlot + 1);
    }

    private void resetCurrentZoneStartSlot() {
        if ("partition1".equals(currentZoneKey)) {
            nextSlotByZone.put(currentZoneKey, 1);
        } else if ("partition2".equals(currentZoneKey)) {
            nextSlotByZone.put(currentZoneKey, 42);
        } else {
            nextSlotByZone.put(currentZoneKey, 67);
        }
    }

    private void notifySlotChanged() {
        if (listener != null) {
            listener.onSlotChanged(getCurrentSlots().size(), currentZoneLabel, getSelectedSlotNumber());
        }
    }

    private float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private static class ParkingSlot {
        private final int slotNumber;
        private float cx;
        private float cy;
        private float w;
        private float h;
        private int angle;
        private boolean disabled;

        private ParkingSlot(int slotNumber, float cx, float cy, float w, float h, int angle, boolean disabled) {
            this.slotNumber = slotNumber;
            this.cx = cx;
            this.cy = cy;
            this.w = w;
            this.h = h;
            this.angle = angle;
            this.disabled = disabled;
        }
    }
}

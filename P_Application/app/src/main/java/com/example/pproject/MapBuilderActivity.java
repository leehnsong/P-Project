package com.example.pproject;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.os.Bundle;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.RadioGroup;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONException;

import java.util.Locale;

public class MapBuilderActivity extends AppCompatActivity {

    private MapBuilderView mapBuilderView;
    private TextView tvBuilderStatus;
    private TextView tvBuilderOutput;
    private TextView tvAngleValue;
    private TextView tvWidthValue;
    private TextView tvHeightValue;
    private CheckBox cbDisabledSlot;
    private SeekBar seekAngle;
    private SeekBar seekWidth;
    private SeekBar seekHeight;
    private boolean syncingControls;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_map_builder);

        mapBuilderView = findViewById(R.id.mapBuilderView);
        tvBuilderStatus = findViewById(R.id.tvBuilderStatus);
        tvBuilderOutput = findViewById(R.id.tvBuilderOutput);
        tvAngleValue = findViewById(R.id.tvAngleValue);
        tvWidthValue = findViewById(R.id.tvWidthValue);
        tvHeightValue = findViewById(R.id.tvHeightValue);
        cbDisabledSlot = findViewById(R.id.cbDisabledSlot);
        seekAngle = findViewById(R.id.seekAngle);
        seekWidth = findViewById(R.id.seekWidth);
        seekHeight = findViewById(R.id.seekHeight);
        RadioGroup rgBuilderZone = findViewById(R.id.rgBuilderZone);
        Button btnUndoSlot = findViewById(R.id.btnUndoSlot);
        Button btnDeleteSlot = findViewById(R.id.btnDeleteSlot);
        Button btnClearZone = findViewById(R.id.btnClearZone);
        Button btnExportJson = findViewById(R.id.btnExportJson);
        Button btnCopyJson = findViewById(R.id.btnCopyJson);

        configureSeekBars();

        mapBuilderView.setOnSlotChangedListener((slotCount, zoneLabel, selectedSlotNumber) -> {
            updateStatus(slotCount);
            syncControlValues();
        });
        mapBuilderView.setZone("partition1", "P1", R.drawable.map_p1);

        rgBuilderZone.setOnCheckedChangeListener((group, checkedId) -> {
            if (checkedId == R.id.rbBuilderP1) {
                mapBuilderView.setZone("partition1", "P1", R.drawable.map_p1);
            } else if (checkedId == R.id.rbBuilderP2) {
                mapBuilderView.setZone("partition2", "P2", R.drawable.map_p2);
            } else if (checkedId == R.id.rbBuilderP3) {
                mapBuilderView.setZone("partition3", "P3", R.drawable.map_p3);
            }
        });

        cbDisabledSlot.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (!syncingControls) {
                mapBuilderView.setSelectedSlotDisabled(isChecked);
            }
        });
        btnUndoSlot.setOnClickListener(v -> mapBuilderView.undoLastSlot());
        btnDeleteSlot.setOnClickListener(v -> mapBuilderView.deleteSelectedSlot());
        btnClearZone.setOnClickListener(v -> mapBuilderView.clearCurrentZone());
        btnExportJson.setOnClickListener(v -> exportJsonToTextView());
        btnCopyJson.setOnClickListener(v -> copyJsonToClipboard());
    }

    private void configureSeekBars() {
        seekAngle.setMax(360);
        seekWidth.setMax(200);
        seekHeight.setMax(200);

        seekAngle.setOnSeekBarChangeListener(new SimpleSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                tvAngleValue.setText(String.format(Locale.KOREA, "%d도", progress));
                if (fromUser && !syncingControls) {
                    mapBuilderView.setSelectedAngle(progress);
                }
            }
        });

        seekWidth.setOnSeekBarChangeListener(new SimpleSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                int value = Math.max(5, progress);
                tvWidthValue.setText(String.format(Locale.KOREA, "%d", value));
                if (fromUser && !syncingControls) {
                    mapBuilderView.setSelectedWidth(value);
                }
            }
        });

        seekHeight.setOnSeekBarChangeListener(new SimpleSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                int value = Math.max(5, progress);
                tvHeightValue.setText(String.format(Locale.KOREA, "%d", value));
                if (fromUser && !syncingControls) {
                    mapBuilderView.setSelectedHeight(value);
                }
            }
        });
    }

    private void syncControlValues() {
        syncingControls = true;
        seekAngle.setProgress(mapBuilderView.getSelectedAngle());
        seekWidth.setProgress(mapBuilderView.getSelectedWidth());
        seekHeight.setProgress(mapBuilderView.getSelectedHeight());
        cbDisabledSlot.setChecked(mapBuilderView.isSelectedSlotDisabled());
        tvAngleValue.setText(String.format(Locale.KOREA, "%d도", mapBuilderView.getSelectedAngle()));
        tvWidthValue.setText(String.format(Locale.KOREA, "%d", mapBuilderView.getSelectedWidth()));
        tvHeightValue.setText(String.format(Locale.KOREA, "%d", mapBuilderView.getSelectedHeight()));
        syncingControls = false;
    }

    private void updateStatus(int slotCount) {
        tvBuilderStatus.setText(mapBuilderView.getCurrentNextSlotLabel() + " / 등록: " + slotCount + "개");
    }

    private void exportJsonToTextView() {
        try {
            tvBuilderOutput.setText(mapBuilderView.exportMappingJson());
        } catch (JSONException e) {
            Toast.makeText(this, "JSON 생성 실패: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private void copyJsonToClipboard() {
        try {
            String json = mapBuilderView.exportMappingJson();
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (clipboard != null) {
                clipboard.setPrimaryClip(ClipData.newPlainText("mapping_parking_slot", json));
                tvBuilderOutput.setText(json);
                Toast.makeText(this, "맵 JSON을 클립보드에 복사했습니다.", Toast.LENGTH_SHORT).show();
            }
        } catch (JSONException e) {
            Toast.makeText(this, "JSON 복사 실패: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private abstract static class SimpleSeekBarChangeListener implements SeekBar.OnSeekBarChangeListener {
        @Override
        public void onStartTrackingTouch(SeekBar seekBar) {
        }

        @Override
        public void onStopTrackingTouch(SeekBar seekBar) {
        }
    }
}

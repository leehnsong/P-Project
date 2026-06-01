package com.example.pproject;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.os.Bundle;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONException;

public class MapBuilderActivity extends AppCompatActivity {

    private MapBuilderView mapBuilderView;
    private TextView tvBuilderStatus;
    private TextView tvBuilderOutput;
    private CheckBox cbDisabledSlot;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_map_builder);

        mapBuilderView = findViewById(R.id.mapBuilderView);
        tvBuilderStatus = findViewById(R.id.tvBuilderStatus);
        tvBuilderOutput = findViewById(R.id.tvBuilderOutput);
        cbDisabledSlot = findViewById(R.id.cbDisabledSlot);
        RadioGroup rgBuilderZone = findViewById(R.id.rgBuilderZone);
        Button btnUndoSlot = findViewById(R.id.btnUndoSlot);
        Button btnClearZone = findViewById(R.id.btnClearZone);
        Button btnExportJson = findViewById(R.id.btnExportJson);
        Button btnCopyJson = findViewById(R.id.btnCopyJson);

        mapBuilderView.setOnSlotChangedListener((slotCount, currentZoneLabel) -> updateStatus(slotCount));
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

        cbDisabledSlot.setOnCheckedChangeListener((buttonView, isChecked) -> mapBuilderView.setNextSlotDisabled(isChecked));
        btnUndoSlot.setOnClickListener(v -> mapBuilderView.undoLastSlot());
        btnClearZone.setOnClickListener(v -> mapBuilderView.clearCurrentZone());
        btnExportJson.setOnClickListener(v -> exportJsonToTextView());
        btnCopyJson.setOnClickListener(v -> copyJsonToClipboard());
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
}

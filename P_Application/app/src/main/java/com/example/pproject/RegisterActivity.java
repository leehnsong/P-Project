package com.example.pproject;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import com.example.pproject.dto.LoginRequest;
import com.example.pproject.network.ApiService;
import com.example.pproject.network.RetrofitClient;
import com.google.android.material.textfield.TextInputEditText;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class RegisterActivity extends AppCompatActivity {

    TextInputEditText editRegId, editRegPw;
    Button btnRegisterAction;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        editRegId = findViewById(R.id.editRegId);
        editRegPw = findViewById(R.id.editRegPw);
        btnRegisterAction = findViewById(R.id.btnRegisterAction);

        btnRegisterAction.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String id = editRegId.getText().toString().trim();
                String pw = editRegPw.getText().toString().trim();

                if (id.isEmpty() || pw.isEmpty()) {
                    Toast.makeText(RegisterActivity.this, "정보를 입력해주세요.", Toast.LENGTH_SHORT).show();
                    return;
                }

                // 서버 통신 시작
                ApiService apiService = RetrofitClient.getApiService();
                LoginRequest req = new LoginRequest(id, pw);

                apiService.register(req).enqueue(new Callback<String>() {
                    @Override
                    public void onResponse(Call<String> call, Response<String> response) {
                        if (response.isSuccessful() && response.body() != null) {
                            String result = response.body();
                            if (result.equals("REGISTER_SUCCESS")) {
                                Toast.makeText(RegisterActivity.this, "가입 성공!", Toast.LENGTH_SHORT).show();
                                finish();
                            } else if (result.equals("USER_EXISTS")) {
                                Toast.makeText(RegisterActivity.this, "이미 존재하는 ID입니다.", Toast.LENGTH_SHORT).show();
                            } else {
                                Toast.makeText(RegisterActivity.this, "가입 실패: " + result, Toast.LENGTH_SHORT).show();
                            }
                        } else {
                            Toast.makeText(RegisterActivity.this, "서버 오류", Toast.LENGTH_SHORT).show();
                        }
                    }

                    @Override
                    public void onFailure(Call<String> call, Throwable t) {
                        Toast.makeText(RegisterActivity.this, "통신 실패: " + t.getMessage(), Toast.LENGTH_SHORT).show();
                    }
                });
            }
        });
    }
}
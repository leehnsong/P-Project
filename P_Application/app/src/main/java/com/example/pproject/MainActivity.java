package com.example.pproject;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import com.example.pproject.dto.LoginRequest;
import com.example.pproject.network.ApiService;
import com.example.pproject.network.RetrofitClient;
import com.google.android.material.textfield.TextInputEditText;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends AppCompatActivity {

    TextInputEditText editId, editPw;
    Button btnLogin;
    TextView tvRegister;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        editId = findViewById(R.id.editId);
        editPw = findViewById(R.id.editPw);
        btnLogin = findViewById(R.id.btnLogin);
        tvRegister = findViewById(R.id.tvRegister);

        tvRegister.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, RegisterActivity.class);
            startActivity(intent);
        });

        btnLogin.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String inputId = editId.getText().toString().trim();
                String inputPw = editPw.getText().toString().trim();

                if(inputId.isEmpty() || inputPw.isEmpty()) {
                    Toast.makeText(MainActivity.this, "ID와 PW를 입력하세요.", Toast.LENGTH_SHORT).show();
                    return;
                }

                // 서버 로그인 요청
                ApiService apiService = RetrofitClient.getApiService();
                LoginRequest req = new LoginRequest(inputId, inputPw);

                apiService.login(req).enqueue(new Callback<String>() {
                    @Override
                    public void onResponse(Call<String> call, Response<String> response) {
                        if (response.isSuccessful() && response.body() != null) {
                            String result = response.body();

                            // 서버에서 실패 시 단순 문자열을 리턴하므로 체크
                            if(result.equals("USER_NOT_FOUND") || result.equals("WRONG_PASSWORD")) {
                                Toast.makeText(MainActivity.this, "로그인 실패: " + result, Toast.LENGTH_SHORT).show();
                            } else {
                                // 성공 시 JWT 토큰이 옴 (토큰 저장)
                                SharedPreferences sharedPreferences = getSharedPreferences("AuthData", MODE_PRIVATE);
                                SharedPreferences.Editor editor = sharedPreferences.edit();
                                editor.putString("jwt_token", result);
                                editor.putString("username", inputId);
                                editor.apply();

                                Toast.makeText(MainActivity.this, inputId + "님 환영합니다!", Toast.LENGTH_SHORT).show();

                                Intent intent = new Intent(MainActivity.this, HomeActivity.class);
                                startActivity(intent);
                                finish();
                            }
                        } else {
                            Toast.makeText(MainActivity.this, "서버 응답 오류", Toast.LENGTH_SHORT).show();
                        }
                    }

                    @Override
                    public void onFailure(Call<String> call, Throwable t) {
                        Toast.makeText(MainActivity.this, "네트워크 오류: " + t.getMessage(), Toast.LENGTH_SHORT).show();
                    }
                });
            }
        });
    }
}
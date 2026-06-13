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
import com.example.pproject.dto.LoginResponse;
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

                apiService.login(req).enqueue(new Callback<LoginResponse>() {
                    @Override
                    public void onResponse(Call<LoginResponse> call, Response<LoginResponse> response) {
                        if (response.isSuccessful() && response.body() != null
                                && response.body().token != null) {
                            // 성공: JSON {token, username} 에서 토큰만 저장
                            String token = response.body().token;
                            String username = response.body().username != null
                                    ? response.body().username : inputId;

                            SharedPreferences sharedPreferences = getSharedPreferences("AuthData", MODE_PRIVATE);
                            SharedPreferences.Editor editor = sharedPreferences.edit();
                            editor.putString("jwt_token", token);
                            editor.putString("username", username);
                            editor.apply();

                            Toast.makeText(MainActivity.this, username + "님 환영합니다!", Toast.LENGTH_SHORT).show();

                            Intent intent = new Intent(MainActivity.this, HomeActivity.class);
                            startActivity(intent);
                            finish();
                        } else {
                            // 401 등: 자격 증명 오류
                            Toast.makeText(MainActivity.this, "로그인 실패: 아이디/비밀번호를 확인하세요.", Toast.LENGTH_SHORT).show();
                        }
                    }

                    @Override
                    public void onFailure(Call<LoginResponse> call, Throwable t) {
                        Toast.makeText(MainActivity.this, "네트워크 오류: " + t.getMessage(), Toast.LENGTH_SHORT).show();
                    }
                });
            }
        });
    }
}
package com.example.pproject;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.pproject.dto.VoiceAskRequest;
import com.example.pproject.dto.VoiceAskResponse;
import com.example.pproject.network.RetrofitClient;

import java.util.ArrayList;
import java.util.Locale;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

/**
 * 음성으로 주차 현황을 묻는 화면.
 * 흐름: 마이크 탭 → SpeechRecognizer(ko-KR) → POST /api/voice/ask → 답변 표시 + TextToSpeech 발화.
 * STT/TTS는 안드로이드 네이티브, 답변 생성은 백엔드(Gemini). 인증 불필요.
 */
public class VoiceActivity extends AppCompatActivity {

    private static final int REQ_RECORD_AUDIO = 1001;

    private SpeechRecognizer speechRecognizer;
    private TextToSpeech tts;
    private boolean ttsReady = false;

    private CardView btnMic;
    private TextView tvStatus;
    private TextView tvQuestion;
    private TextView tvAnswer;

    private enum State { IDLE, LISTENING, ASKING, SPEAKING }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_voice);

        btnMic = findViewById(R.id.btnMic);
        tvStatus = findViewById(R.id.tvVoiceStatus);
        tvQuestion = findViewById(R.id.tvVoiceQuestion);
        tvAnswer = findViewById(R.id.tvVoiceAnswer);

        tts = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                tts.setLanguage(Locale.KOREAN);
                tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                    @Override public void onStart(String utteranceId) { }
                    @Override public void onDone(String utteranceId) { runOnUiThread(() -> setState(State.IDLE)); }
                    @Override public void onError(String utteranceId) { runOnUiThread(() -> setState(State.IDLE)); }
                });
                ttsReady = true;
            }
        });

        btnMic.setOnClickListener(v -> onMicTapped());
        setState(State.IDLE);
    }

    private void onMicTapped() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO}, REQ_RECORD_AUDIO);
            return;
        }
        startListening();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_RECORD_AUDIO) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startListening();
            } else {
                Toast.makeText(this, "마이크 권한이 필요합니다.", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void startListening() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            Toast.makeText(this, "이 기기에서 음성 인식을 사용할 수 없어요.", Toast.LENGTH_SHORT).show();
            return;
        }
        if (speechRecognizer == null) {
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);
            speechRecognizer.setRecognitionListener(recognitionListener);
        }
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR");
        tvQuestion.setText("");
        tvAnswer.setText("");
        setState(State.LISTENING);
        speechRecognizer.startListening(intent);
    }

    private final RecognitionListener recognitionListener = new RecognitionListener() {
        @Override public void onReadyForSpeech(Bundle params) { }
        @Override public void onBeginningOfSpeech() { }
        @Override public void onRmsChanged(float rmsdB) { }
        @Override public void onBufferReceived(byte[] buffer) { }
        @Override public void onEndOfSpeech() { }
        @Override public void onEvent(int eventType, Bundle params) { }
        @Override public void onPartialResults(Bundle partialResults) { }

        @Override
        public void onError(int error) {
            setState(State.IDLE);
            tvStatus.setText("잘 못 들었어요. 마이크를 다시 눌러 말씀해 주세요.");
        }

        @Override
        public void onResults(Bundle results) {
            ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
            if (matches == null || matches.isEmpty() || matches.get(0).trim().isEmpty()) {
                setState(State.IDLE);
                tvStatus.setText("잘 못 들었어요. 다시 시도해 주세요.");
                return;
            }
            String question = matches.get(0);
            tvQuestion.setText("질문: " + question);
            askBackend(question);
        }
    };

    private void askBackend(String question) {
        setState(State.ASKING);
        RetrofitClient.getApiService()
                .askVoice(new VoiceAskRequest(question))
                .enqueue(new Callback<VoiceAskResponse>() {
                    @Override
                    public void onResponse(@NonNull Call<VoiceAskResponse> call,
                                           @NonNull Response<VoiceAskResponse> response) {
                        if (response.isSuccessful() && response.body() != null
                                && response.body().getAnswer() != null) {
                            String answer = response.body().getAnswer();
                            tvAnswer.setText(answer);
                            speak(answer);
                        } else {
                            setState(State.IDLE);
                            tvAnswer.setText("답변을 받지 못했어요.");
                        }
                    }

                    @Override
                    public void onFailure(@NonNull Call<VoiceAskResponse> call, @NonNull Throwable t) {
                        setState(State.IDLE);
                        tvAnswer.setText("네트워크 오류로 답변을 받지 못했어요. 백엔드 서버가 켜져 있는지 확인해 주세요.");
                    }
                });
    }

    private void speak(String text) {
        if (ttsReady && tts != null) {
            setState(State.SPEAKING);
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "voice-answer");
        } else {
            setState(State.IDLE);
        }
    }

    private void setState(State state) {
        switch (state) {
            case IDLE:
                tvStatus.setText("마이크를 누르고 말해보세요");
                break;
            case LISTENING:
                tvStatus.setText("🎙️ 듣는 중...");
                break;
            case ASKING:
                tvStatus.setText("답변 생성 중...");
                break;
            case SPEAKING:
                tvStatus.setText("🔊 답변 중...");
                break;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (speechRecognizer != null) {
            speechRecognizer.destroy();
        }
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
    }
}

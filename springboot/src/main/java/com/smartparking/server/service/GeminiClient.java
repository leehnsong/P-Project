package com.smartparking.server.service;

import com.smartparking.server.dto.GeminiResponse;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Slf4j
@Service
public class GeminiClient {

    private static final String MODEL = "gemini-2.5-flash-lite";

    private final WebClient geminiWebClient;
    private final String apiKey;

    public GeminiClient(WebClient geminiWebClient,
                        @Value("${smartparking.gemini.api-key:}") String apiKey) {
        this.geminiWebClient = geminiWebClient;
        this.apiKey = apiKey;
    }

    /** 프롬프트를 보내고 생성된 텍스트를 반환. 실패 시 null. */
    public String generate(String prompt) {
        if (apiKey == null || apiKey.isBlank()) {
            log.warn("Gemini API 키가 설정되지 않음");
            return null;
        }

        Map<String, Object> body = Map.of(
                "contents", List.of(Map.of("parts", List.of(Map.of("text", prompt)))),
                "generationConfig", Map.of("maxOutputTokens", 100, "temperature", 0.3));

        try {
            GeminiResponse response = geminiWebClient.post()
                    .uri(uriBuilder -> uriBuilder
                            .path("/v1beta/models/" + MODEL + ":generateContent")
                            .queryParam("key", apiKey)
                            .build())
                    .bodyValue(body)
                    .retrieve()
                    .bodyToMono(GeminiResponse.class)
                    .timeout(Duration.ofSeconds(8))
                    .block();

            if (response == null || response.getCandidates() == null || response.getCandidates().isEmpty()) {
                return null;
            }
            GeminiResponse.Content content = response.getCandidates().get(0).getContent();
            if (content == null || content.getParts() == null || content.getParts().isEmpty()) {
                return null;
            }
            String text = content.getParts().get(0).getText();
            return text == null ? null : text.trim();
        } catch (Exception e) {
            log.warn("Gemini 호출 실패: {}", e.getMessage());
            return null;
        }
    }
}

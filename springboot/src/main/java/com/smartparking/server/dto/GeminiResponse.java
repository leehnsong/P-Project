package com.smartparking.server.dto;

import java.util.List;
import lombok.Data;

/** Gemini generateContent 응답에서 필요한 부분만 매핑. (알 수 없는 필드는 무시됨) */
@Data
public class GeminiResponse {

    private List<Candidate> candidates;

    @Data
    public static class Candidate {
        private Content content;
    }

    @Data
    public static class Content {
        private List<Part> parts;
    }

    @Data
    public static class Part {
        private String text;
    }
}

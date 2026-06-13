package com.example.pproject.dto;

public class InAppNotificationResponse {
    public Long id;
    public String title;
    public String message;
    public String category;
    public boolean read;
    public String createdAt; // ISO-8601 문자열
    public String readAt;
}

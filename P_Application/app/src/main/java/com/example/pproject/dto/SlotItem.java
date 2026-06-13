package com.example.pproject.dto;

import java.util.List;

public class SlotItem {
    public Integer slotId;
    public String type;       // "normal" | "disabled"
    public String status;     // "available" | "occupied"
    public List<Double> center; // [x, y]
}

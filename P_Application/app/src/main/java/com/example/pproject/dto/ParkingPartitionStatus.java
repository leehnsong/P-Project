package com.example.pproject.dto;

import java.util.List;

public class ParkingPartitionStatus {
    public List<Integer> occupied_slots;
    public List<Integer> available_slots;
    public List<Integer> disabled_slots;
    public Integer total_available;
}

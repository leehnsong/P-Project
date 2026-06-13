package com.smartparking.server.repository;

import com.smartparking.server.entity.SavedParkingLocation;
import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.transaction.annotation.Transactional;

public interface SavedParkingLocationRepository extends JpaRepository<SavedParkingLocation, Long> {
    Optional<SavedParkingLocation> findFirstByUserUsernameAndActiveTrueOrderBySavedAtDesc(String username);

    List<SavedParkingLocation> findByUserUsernameOrderBySavedAtDesc(String username);

    List<SavedParkingLocation> findByUserUsernameAndActiveTrue(String username);

    @Transactional
    void deleteByParkingLotId(Long parkingLotId);
}

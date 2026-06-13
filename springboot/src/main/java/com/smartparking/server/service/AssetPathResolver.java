package com.smartparking.server.service;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class AssetPathResolver {

    private final String configuredRoot;

    public AssetPathResolver(@Value("${smartparking.asset-root:}") String configuredRoot) {
        this.configuredRoot = configuredRoot;
    }

    public Path videoTestRoot() {
        if (configuredRoot != null && !configuredRoot.isBlank()) {
            return Paths.get(configuredRoot).toAbsolutePath().normalize();
        }
        return resolveExistingPath(
                Paths.get("fastapi", "video_test"),
                Paths.get("..", "fastapi", "video_test"),
                Paths.get("..", "..", "fastapi", "video_test"));
    }

    public Path videoPath(String partitionKey) {
        return videoTestRoot().resolve("videos").resolve(partitionKey + "_video.mp4");
    }

    public Path sourceImagePath(String partitionKey) {
        return videoTestRoot().resolve("images").resolve(partitionKey + "_image.png");
    }

    public Path generatedMapPath(String partitionKey) {
        return videoTestRoot().resolve("map").resolve(partitionKey + "_map.png");
    }

    public Path slotLayoutPath(String partitionKey) {
        return videoTestRoot().resolve("map").resolve(partitionKey + "_slots.json");
    }

    private Path resolveExistingPath(Path... candidates) {
        for (Path candidate : candidates) {
            Path absolute = candidate.toAbsolutePath().normalize();
            if (Files.exists(absolute)) {
                return absolute;
            }
        }
        return candidates[0].toAbsolutePath().normalize();
    }
}

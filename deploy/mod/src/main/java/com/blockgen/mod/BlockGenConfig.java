package com.blockgen.mod;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import net.fabricmc.loader.api.FabricLoader;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Persisted settings, at {@code config/blockgen.json}.
 *
 * <p>The inference server runs on a different machine from Minecraft (that is the
 * whole point: GPU box vs laptop), so the URL has to be configurable at runtime via
 * {@code /blockgen server <url>} rather than baked in. Everything else here is a
 * demo-pacing knob.
 */
public class BlockGenConfig {
    /** WebSocket endpoint of the inference server. */
    public String serverUrl = "ws://127.0.0.1:8000/ws";
    /** Model name, or null to use whatever the server calls default. */
    public String model = null;
    /**
     * Blocks placed per server tick. At 20 tps the default is ~640 blocks/s, which
     * outruns the model (~170 blocks/s for native_bpe) so the build appears at the
     * model's pace rather than the placer's. Lower it to watch, raise it to hurry.
     */
    public int blocksPerTick = 32;
    /** Sampling temperature passed through to the server. */
    public double temperature = 1.0;
    /** Top-k passed through to the server. */
    public int topK = 40;
    /** Classifier-free guidance scale, used only by text-conditioned models. */
    public double cfgScale = 3.0;
    /** Place the build this many blocks above the player's feet. */
    public int yOffset = 0;
    /** How many past builds {@code /blockgen undo} can walk back. */
    public int undoHistory = 5;

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private static BlockGenConfig instance;

    public static Path path() {
        return FabricLoader.getInstance().getConfigDir().resolve("blockgen.json");
    }

    public static BlockGenConfig get() {
        if (instance == null) {
            instance = load();
        }
        return instance;
    }

    private static BlockGenConfig load() {
        Path p = path();
        if (Files.exists(p)) {
            try {
                BlockGenConfig c = GSON.fromJson(Files.readString(p), BlockGenConfig.class);
                if (c != null) {
                    return c;
                }
            } catch (IOException | RuntimeException e) {
                // A hand-edited config that fails to parse should not stop the mod from
                // loading; fall back to defaults and say so.
                BlockGenMod.LOGGER.warn("Could not read {}, using defaults: {}", p, e.toString());
            }
        }
        return new BlockGenConfig();
    }

    public void save() {
        try {
            Files.createDirectories(path().getParent());
            Files.writeString(path(), GSON.toJson(this));
        } catch (IOException e) {
            BlockGenMod.LOGGER.error("Could not save config to {}", path(), e);
        }
    }
}

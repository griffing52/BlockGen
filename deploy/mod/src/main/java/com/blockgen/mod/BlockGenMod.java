package com.blockgen.mod;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.minecraft.block.BlockState;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.math.BlockPos;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Mod entrypoint: owns the per-player sessions and drives them from the server tick.
 *
 * <p>Sessions are keyed by player UUID, so two players on one server generate into
 * their own anchors without interleaving. Everything in here runs on the server
 * thread; see {@link GenerationSession} for the network/tick boundary.
 */
public class BlockGenMod implements ModInitializer {
    public static final String MOD_ID = "blockgen";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    private static final Map<UUID, GenerationSession> SESSIONS = new ConcurrentHashMap<>();
    private static final Deque<UndoRecord> UNDO = new ArrayDeque<>();

    /** A completed build's original block states, so it can be reverted. */
    private record UndoRecord(ServerWorld world, Map<BlockPos, BlockState> previous) {}

    @Override
    public void onInitialize() {
        CommandRegistrationCallback.EVENT.register(
                (dispatcher, registryAccess, environment) -> BlockGenCommands.register(dispatcher));

        ServerTickEvents.END_SERVER_TICK.register(server -> {
            SESSIONS.entrySet().removeIf(e -> e.getValue().tick());
        });

        // A world unload with a live session would leave the placer writing into a world
        // that is going away; drop everything rather than risk it.
        ServerLifecycleEvents.SERVER_STOPPING.register(server -> {
            SESSIONS.values().forEach(GenerationSession::cancel);
            SESSIONS.clear();
            UNDO.clear();
        });

        LOGGER.info("BlockGen ready. /gen to build, /model to list models, "
                + "/blockgen server <url> to point at your inference server.");
    }

    public static GenerationSession session(UUID player) {
        return SESSIONS.get(player);
    }

    public static void putSession(UUID player, GenerationSession s) {
        SESSIONS.put(player, s);
    }

    public static boolean isBusy(UUID player) {
        return SESSIONS.containsKey(player);
    }

    static void pushUndo(ServerWorld world, Map<BlockPos, BlockState> previous, int limit) {
        if (previous.isEmpty()) {
            return;
        }
        UNDO.push(new UndoRecord(world, Map.copyOf(previous)));
        while (UNDO.size() > Math.max(1, limit)) {
            UNDO.removeLast();
        }
    }

    /** Reverts the most recent build. Returns blocks restored, or -1 if there is none. */
    public static int undo() {
        UndoRecord rec = UNDO.poll();
        if (rec == null) {
            return -1;
        }
        for (Map.Entry<BlockPos, BlockState> e : rec.previous().entrySet()) {
            rec.world().setBlockState(e.getKey(), e.getValue());
        }
        return rec.previous().size();
    }
}

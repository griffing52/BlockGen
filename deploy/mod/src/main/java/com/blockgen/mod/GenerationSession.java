package com.blockgen.mod;

import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.command.argument.BlockArgumentParser;
import net.minecraft.registry.RegistryKeys;
import net.minecraft.server.command.ServerCommandSource;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.text.Text;
import net.minecraft.util.Formatting;
import net.minecraft.util.math.BlockPos;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * One in-flight generation: receives blocks off the network and places them in world.
 *
 * <p><b>The thread boundary is the whole design.</b> Blocks arrive on WebSocket
 * threads; the world may only be touched on the server thread. So the network side
 * only ever enqueues, and {@link #tick} drains the queue during the server tick.
 * Placing directly from the network callback would race the tick loop and corrupt
 * chunks -- intermittently, which is the worst way to find out.
 *
 * <p>Placement is rate-limited per tick rather than drained greedily, because the
 * point is to <em>watch</em> the model build. The queue also decouples the two rates:
 * the model streams at ~170 blocks/s, so a 32/tick budget (~640/s) keeps the world
 * at the model's pace and the queue near-empty.
 */
public class GenerationSession {

    private final ServerCommandSource source;
    private final ServerWorld world;
    private final BlockPos anchor;
    private final BlockGenConfig cfg;
    private final InferenceClient client = new InferenceClient();

    private final ConcurrentLinkedQueue<InferenceClient.PendingBlock> incoming =
            new ConcurrentLinkedQueue<>();
    /** Original states, for undo. LinkedHashMap keeps first-write-wins per position. */
    private final Map<BlockPos, BlockState> previous = new LinkedHashMap<>();

    private final AtomicBoolean finished = new AtomicBoolean(false);
    private final AtomicBoolean cancelled = new AtomicBoolean(false);
    private volatile String doneMessage = null;
    private volatile String modelName = "?";
    private volatile long seed = -1;
    private int placed = 0;
    private int unknownStates = 0;
    private final Map<String, BlockState> stateCache = new HashMap<>();

    /**
     * @param source where feedback goes; a player for {@code /gen}, or the console
     *               for {@code /blockgen at}, which is also what makes this testable
     *               headlessly and usable from command blocks.
     * @param world  the world to build in -- passed explicitly rather than taken from
     *               the source, which has no world when it is the console.
     * @param anchor the build's origin; model coordinates are added to it.
     */
    public GenerationSession(ServerCommandSource source, ServerWorld world,
                             BlockPos anchor, BlockGenConfig cfg) {
        this.source = source;
        this.world = world;
        this.anchor = anchor;
        this.cfg = cfg;
    }

    public void start(String model, String prompt, Long seedIn) {
        feedback(Text.literal("Generating" + (prompt != null ? " \"" + prompt + "\"" : "")
                + "...").formatted(Formatting.GRAY));
        client.generate(cfg.serverUrl, model, prompt, seedIn, cfg, new InferenceClient.Handler() {
            @Override
            public void onBegin(String m, long s, String p) {
                modelName = m;
                seed = s;
            }

            @Override
            public void onBlocks(List<InferenceClient.PendingBlock> blocks) {
                incoming.addAll(blocks);
            }

            @Override
            public void onDone(int blocks, double elapsed, String reason) {
                doneMessage = switch (reason) {
                    case "cancelled" -> "Cancelled after " + blocks + " blocks.";
                    default -> String.format("Done: %d blocks in %.1fs from %s (seed %d). "
                            + "/blockgen undo to remove.", blocks, elapsed, modelName, seed);
                };
                finished.set(true);
            }

            @Override
            public void onError(String message) {
                doneMessage = null;
                // Errors are reported from tick() so the message ordering matches the
                // placement it refers to, and so Text is built on the server thread.
                errorMessage = message;
                finished.set(true);
            }
        });
    }

    private volatile String errorMessage = null;

    /** Called every server tick. Returns true when the session is over. */
    public boolean tick() {
        int budget = Math.max(1, cfg.blocksPerTick);
        InferenceClient.PendingBlock b;
        while (budget-- > 0 && (b = incoming.poll()) != null) {
            place(b);
        }
        boolean drained = incoming.isEmpty();
        if (finished.get() && drained) {
            if (errorMessage != null) {
                feedback(Text.literal("BlockGen: " + errorMessage).formatted(Formatting.RED));
            } else if (doneMessage != null) {
                String extra = unknownStates > 0
                        ? " (" + unknownStates + " unknown block states skipped)" : "";
                feedback(Text.literal(doneMessage + extra).formatted(Formatting.GREEN));
            }
            if (placed > 0) {
                BlockGenMod.pushUndo(world, previous, cfg.undoHistory);
            }
            client.close();
            return true;
        }
        return false;
    }

    private void place(InferenceClient.PendingBlock b) {
        BlockState state = resolve(b.state());
        if (state == null) {
            unknownStates++;
            return;
        }
        BlockPos pos = anchor.add(b.x(), b.y() + cfg.yOffset, b.z());
        if (!world.isInBuildLimit(pos)) {
            return;
        }
        previous.putIfAbsent(pos, world.getBlockState(pos));
        world.setBlockState(pos, state);
        placed++;
    }

    /**
     * Parse "minecraft:oak_stairs[facing=east]" into a BlockState.
     *
     * <p>Returns null instead of throwing on an unknown block: the server's block map
     * is built against a specific Minecraft version, so a mismatch should degrade to
     * "that block is missing" rather than abort a build halfway. Results are cached
     * because a build repeats a handful of states thousands of times.
     */
    private BlockState resolve(String spec) {
        BlockState cached = stateCache.get(spec);
        if (cached != null) {
            return cached;
        }
        try {
            BlockState parsed = BlockArgumentParser
                    .block(world.createCommandRegistryWrapper(RegistryKeys.BLOCK), spec, false)
                    .blockState();
            stateCache.put(spec, parsed);
            return parsed;
        } catch (Exception e) {
            BlockGenMod.LOGGER.warn("Unknown block state from server: {} ({})", spec, e.toString());
            stateCache.put(spec, Blocks.AIR.getDefaultState());
            return null;
        }
    }

    public void cancel() {
        cancelled.set(true);
        client.cancel();
    }

    public String getModelName() {
        return modelName;
    }

    public int queued() {
        return incoming.size();
    }

    public int placed() {
        return placed;
    }

    private void feedback(Text text) {
        // Runs on the server thread (tick/command), so this is safe. Not broadcast to
        // ops: a build streams dozens of messages and would spam the whole server.
        source.sendFeedback(() -> text, false);
    }
}

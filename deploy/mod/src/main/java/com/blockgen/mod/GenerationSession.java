package com.blockgen.mod;

import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.command.argument.BlockArgumentParser;
import net.minecraft.registry.RegistryKeys;
import net.minecraft.server.command.ServerCommandSource;
import net.minecraft.server.network.ServerPlayerEntity;
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

    /**
     * Blocks and chat lines, in arrival order. One queue rather than two so a line
     * prints exactly when the blocks before it have been placed: "attempt 2 scored
     * 0.61" should appear as attempt 2 finishes going up, not while attempt 1 is
     * still being placed from a backlog.
     */
    private final ConcurrentLinkedQueue<Object> incoming = new ConcurrentLinkedQueue<>();

    /** A narration line waiting in {@link #incoming}. */
    private record ChatLine(String text, String style) {}
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
            public void onStep(int index, int total, String command, int blocks,
                               String error) {
                // Programs (agentic models) build in named steps; queue the label so
                // it prints from the tick loop, in order with the blocks it describes.
                if (!cfg.showSteps) return;
                String label = String.format("[%d/%d] %s", index, total, command);
                incoming.add(new ChatLine(error != null ? label + "  <- " + error : label,
                        "step"));
            }

            @Override
            public void onLog(String text, String style) {
                // Recipe narration. Stage changes, scores and problems always show;
                // the running commentary (thinking, replies, placements) follows
                // /blockgen steps.
                boolean essential = switch (style) {
                    case "stage", "score", "warn", "error" -> true;
                    default -> false;
                };
                if (essential || cfg.showSteps) incoming.add(new ChatLine(text, style));
            }

            @Override
            public void onStats(com.google.gson.JsonObject s) {
                statsMessage = formatStats(s);
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
    private volatile String statsMessage = null;

    /**
     * One line of build economics for chat.
     *
     * <p>An API model costs real money per build, so the number belongs in front of
     * the person who triggered it. A model missing from the server's price table
     * reports {@code cost_known=false}; saying "cost unknown" is the honest output --
     * printing $0.00 would read as free.
     */
    private static String formatStats(com.google.gson.JsonObject s) {
        StringBuilder sb = new StringBuilder();
        if (s.has("provider")) sb.append(s.get("provider").getAsString());
        if (s.has("commands")) {
            sb.append("  ").append(s.get("commands").getAsInt()).append(" commands");
            if (s.has("failed_commands") && s.get("failed_commands").getAsInt() > 0) {
                sb.append(" (").append(s.get("failed_commands").getAsInt()).append(" failed)");
            }
        }
        if (s.has("prompt_tokens") && s.has("completion_tokens")) {
            sb.append("  ").append(s.get("prompt_tokens").getAsInt()).append("+")
              .append(s.get("completion_tokens").getAsInt()).append(" tokens");
        }
        boolean known = !s.has("cost_known") || s.get("cost_known").getAsBoolean();
        if (s.has("cached") && s.get("cached").getAsBoolean()) {
            sb.append("  cached (no API call)");
        } else if (known && s.has("cost_usd")) {
            sb.append(String.format("  $%.4f", s.get("cost_usd").getAsDouble()));
        } else {
            sb.append("  cost unknown (model not in the price table)");
        }
        // Recipe builds: which attempt won, the session total, where the trace went.
        if (s.has("attempts")) {
            sb.append("  best attempt ");
            sb.append(s.has("best_attempt") && !s.get("best_attempt").isJsonNull()
                    ? s.get("best_attempt").getAsInt() : -1);
            sb.append(" of ").append(s.get("attempts").getAsInt());
            if (s.has("best_score") && !s.get("best_score").isJsonNull()) {
                sb.append(String.format(" (score %.2f)", s.get("best_score").getAsDouble()));
            }
        }
        if (s.has("session_usd") && !s.get("session_usd").isJsonNull()) {
            sb.append(String.format("  session $%.2f", s.get("session_usd").getAsDouble()));
        }
        if (s.has("trace") && !s.get("trace").isJsonNull()) {
            sb.append("\n  trace: ").append(s.get("trace").getAsString());
        }
        return sb.toString();
    }

    /** Called every server tick. Returns true when the session is over. */
    public boolean tick() {
        int budget = Math.max(1, cfg.blocksPerTick);
        // Lines print as the blocks before them land, not all at once at the end:
        // the point of a program build is watching it go up part by part. Lines do
        // not count against the per-tick block budget.
        Object item;
        while (budget > 0 && (item = incoming.poll()) != null) {
            if (item instanceof ChatLine line) {
                say(line);
            } else {
                place((InferenceClient.PendingBlock) item);
                budget--;
            }
        }
        boolean drained = incoming.isEmpty();
        if (finished.get() && drained) {
            if (errorMessage != null) {
                feedback(Text.literal("BlockGen: " + errorMessage).formatted(Formatting.RED));
            } else if (doneMessage != null) {
                String extra = unknownStates > 0
                        ? " (" + unknownStates + " unknown block states skipped)" : "";
                feedback(Text.literal(doneMessage + extra).formatted(Formatting.GREEN));
                if (statsMessage != null) {
                    feedback(Text.literal("  " + statsMessage).formatted(Formatting.GRAY));
                }
            }
            if (placed > 0) {
                BlockGenMod.pushUndo(world, previous, cfg.undoHistory);
            }
            client.close();
            return true;
        }
        return false;
    }

    /**
     * One narration line. "thinking..." goes to the action bar: it is a status, not
     * a record, and a minute-long LLM call should not leave a wall of chat. Every
     * other style is a chat line coloured by what it is.
     */
    private void say(ChatLine line) {
        if (line.style().equals("think")) {
            ServerPlayerEntity player = source.getPlayer();
            if (player != null) {
                player.sendMessage(Text.literal(line.text()).formatted(Formatting.GRAY,
                        Formatting.ITALIC), true);
                return;
            }
        }
        Formatting color = switch (line.style()) {
            case "stage" -> Formatting.GOLD;
            case "build" -> Formatting.AQUA;
            case "critique", "think" -> Formatting.GRAY;
            case "dream" -> Formatting.LIGHT_PURPLE;
            case "score" -> Formatting.GREEN;
            case "warn" -> Formatting.YELLOW;
            case "error" -> Formatting.RED;
            default -> Formatting.DARK_AQUA;
        };
        feedback(Text.literal("  " + line.text()).formatted(color));
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

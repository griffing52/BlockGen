package com.blockgen.mod;

import com.google.gson.JsonObject;
import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.LongArgumentType;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.builder.LiteralArgumentBuilder;
import com.mojang.brigadier.context.CommandContext;
import net.minecraft.command.argument.BlockPosArgumentType;
import net.minecraft.server.command.ServerCommandSource;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.text.Text;
import net.minecraft.util.Formatting;
import net.minecraft.util.math.BlockPos;

import java.util.UUID;

import static net.minecraft.server.command.CommandManager.argument;
import static net.minecraft.server.command.CommandManager.literal;

/**
 * Command tree.
 *
 * <pre>
 *   /gen                      generate with the current model, at your feet
 *   /gen &lt;prompt...&gt;          text-conditioned generation (needs a text model)
 *   /blockgen cancel          stop the running generation
 *   /blockgen undo            revert the last build
 *   /blockgen model           list models and show the current one
 *   /blockgen model &lt;name&gt;    switch model
 *   /blockgen server [url]    show or set the inference server
 *   /blockgen status          connection + session state
 *   /blockgen speed &lt;n&gt;       blocks placed per tick
 * </pre>
 *
 * <p>{@code /gen}, {@code /generate} and {@code /model} are registered as top-level
 * aliases, since those are what you actually type.
 *
 * <p><b>Note on {@code /gen &lt;prompt&gt;}:</b> the prompt is a greedy string, so
 * {@code /gen cancel} builds a house prompted with the word "cancel" rather than
 * cancelling. Control verbs live under {@code /blockgen} precisely so that a prompt
 * can contain any word.
 */
public final class BlockGenCommands {

    private BlockGenCommands() {}

    public static void register(CommandDispatcher<ServerCommandSource> dispatcher) {
        LiteralArgumentBuilder<ServerCommandSource> root = literal("blockgen")
                .requires(BlockGenCommands::allowed)
                .executes(ctx -> status(ctx))
                .then(literal("gen")
                        .executes(ctx -> generate(ctx, null, null))
                        .then(argument("prompt", StringArgumentType.greedyString())
                                .executes(ctx -> generate(ctx,
                                        StringArgumentType.getString(ctx, "prompt"), null))))
                .then(literal("seed")
                        .then(argument("seed", LongArgumentType.longArg())
                                .executes(ctx -> generate(ctx, null,
                                        LongArgumentType.getLong(ctx, "seed")))))
                // Explicit anchor: works from the console and command blocks, where
                // there is no player to build at.
                .then(literal("at")
                        .then(argument("pos", BlockPosArgumentType.blockPos())
                                .executes(ctx -> generateAt(ctx,
                                        BlockPosArgumentType.getBlockPos(ctx, "pos"), null))
                                .then(argument("prompt", StringArgumentType.greedyString())
                                        .executes(ctx -> generateAt(ctx,
                                                BlockPosArgumentType.getBlockPos(ctx, "pos"),
                                                StringArgumentType.getString(ctx, "prompt"))))))
                .then(literal("cancel").executes(BlockGenCommands::cancel))
                .then(literal("undo").executes(BlockGenCommands::undo))
                .then(literal("status").executes(BlockGenCommands::status))
                .then(literal("model")
                        .executes(BlockGenCommands::listModels)
                        .then(argument("name", StringArgumentType.word())
                                .executes(BlockGenCommands::setModel)))
                .then(literal("server")
                        .executes(BlockGenCommands::showServer)
                        .then(argument("url", StringArgumentType.greedyString())
                                .executes(BlockGenCommands::setServer)))
                .then(literal("speed")
                        .then(argument("n", com.mojang.brigadier.arguments.IntegerArgumentType
                                .integer(1, 4096))
                                .executes(BlockGenCommands::setSpeed)));
        dispatcher.register(root);

        // Aliases: /gen [prompt], /generate [prompt], /model [name]
        for (String alias : new String[]{"gen", "generate"}) {
            dispatcher.register(literal(alias)
                    .requires(BlockGenCommands::allowed)
                    .executes(ctx -> generate(ctx, null, null))
                    .then(argument("prompt", StringArgumentType.greedyString())
                            .executes(ctx -> generate(ctx,
                                    StringArgumentType.getString(ctx, "prompt"), null))));
        }
        dispatcher.register(literal("model")
                .requires(BlockGenCommands::allowed)
                .executes(BlockGenCommands::listModels)
                .then(argument("name", StringArgumentType.word())
                        .executes(BlockGenCommands::setModel)));
    }

    /**
     * Ops on a multiplayer server, anyone in singleplayer.
     *
     * <p>Requiring permission level 2 outright would make the mod unusable in a normal
     * singleplayer world, where the player has permission 0 unless cheats are on --
     * and singleplayer is the main way this gets used.
     */
    private static boolean allowed(ServerCommandSource src) {
        return src.hasPermissionLevel(2)
                || (src.getServer() != null && src.getServer().isSingleplayer());
    }

    private static int generate(CommandContext<ServerCommandSource> ctx, String prompt, Long seed) {
        ServerCommandSource src = ctx.getSource();
        ServerPlayerEntity player;
        try {
            player = src.getPlayerOrThrow();
        } catch (Exception e) {
            src.sendError(Text.literal("/gen must be run by a player (it builds at your feet)."));
            return 0;
        }
        if (BlockGenMod.isBusy(player.getUuid())) {
            src.sendError(Text.literal("Already generating. /blockgen cancel to stop it."));
            return 0;
        }
        BlockGenConfig cfg = BlockGenConfig.get();
        GenerationSession session = new GenerationSession(
                src, player.getServerWorld(), player.getBlockPos(), cfg);
        BlockGenMod.putSession(player.getUuid(), session);
        session.start(cfg.model, prompt, seed);
        return 1;
    }

    /** {@code /blockgen at <pos> [prompt]} -- build at explicit coordinates. */
    private static int generateAt(CommandContext<ServerCommandSource> ctx, BlockPos anchor,
                                  String prompt) {
        ServerCommandSource src = ctx.getSource();
        // Console sessions share one key, so two console builds cannot interleave.
        UUID key = sessionKey(src);
        if (BlockGenMod.isBusy(key)) {
            src.sendError(Text.literal("Already generating. /blockgen cancel to stop it."));
            return 0;
        }
        BlockGenConfig cfg = BlockGenConfig.get();
        GenerationSession session = new GenerationSession(src, src.getWorld(), anchor, cfg);
        BlockGenMod.putSession(key, session);
        session.start(cfg.model, prompt, null);  // server picks a seed and reports it
        return 1;
    }

    /** Player UUID, or a fixed key for the console. */
    private static UUID sessionKey(ServerCommandSource src) {
        ServerPlayerEntity p = src.getPlayer();
        return p != null ? p.getUuid() : CONSOLE_SESSION;
    }

    private static final UUID CONSOLE_SESSION =
            UUID.nameUUIDFromBytes("blockgen:console".getBytes(java.nio.charset.StandardCharsets.UTF_8));

    private static int cancel(CommandContext<ServerCommandSource> ctx) {
        ServerCommandSource src = ctx.getSource();
        GenerationSession s = BlockGenMod.session(sessionKey(src));
        if (s == null) {
            src.sendError(Text.literal("Nothing is generating."));
            return 0;
        }
        s.cancel();
        src.sendFeedback(() -> Text.literal("Cancelling...").formatted(Formatting.GRAY), false);
        return 1;
    }

    private static int undo(CommandContext<ServerCommandSource> ctx) {
        int n = BlockGenMod.undo();
        if (n < 0) {
            ctx.getSource().sendError(Text.literal("Nothing to undo."));
            return 0;
        }
        ctx.getSource().sendFeedback(
                () -> Text.literal("Reverted " + n + " blocks.").formatted(Formatting.GREEN), false);
        return 1;
    }

    private static int status(CommandContext<ServerCommandSource> ctx) {
        BlockGenConfig cfg = BlockGenConfig.get();
        ServerCommandSource src = ctx.getSource();
        src.sendFeedback(() -> Text.literal("BlockGen").formatted(Formatting.GOLD), false);
        src.sendFeedback(() -> Text.literal("  server: " + cfg.serverUrl), false);
        src.sendFeedback(() -> Text.literal("  model:  "
                + (cfg.model == null ? "<server default>" : cfg.model)), false);
        src.sendFeedback(() -> Text.literal(String.format(
                "  sampling: temp %.2f, top-k %d, cfg %.1f, %d blocks/tick",
                cfg.temperature, cfg.topK, cfg.cfgScale, cfg.blocksPerTick)), false);
        GenerationSession s = BlockGenMod.session(sessionKey(src));
        if (s != null) {
            src.sendFeedback(() -> Text.literal("  running: " + s.placed()
                    + " placed, " + s.queued() + " queued"), false);
        }
        return 1;
    }

    private static int listModels(CommandContext<ServerCommandSource> ctx) {
        BlockGenConfig cfg = BlockGenConfig.get();
        ServerCommandSource src = ctx.getSource();
        src.sendFeedback(() -> Text.literal("Asking " + cfg.serverUrl + "...")
                .formatted(Formatting.GRAY), false);
        new InferenceClient().listModels(cfg.serverUrl, new ReportingHandler(src),
                msg -> src.getServer().execute(() -> printModels(src, msg, cfg)));
        return 1;
    }

    /** Runs on the server thread (handed over via server.execute). */
    private static void printModels(ServerCommandSource src, JsonObject msg, BlockGenConfig cfg) {
        String serverDefault = msg.get("default").getAsString();
        String current = cfg.model == null ? serverDefault : cfg.model;
        src.sendFeedback(() -> Text.literal("Models (current: " + current + ")")
                .formatted(Formatting.GOLD), false);
        msg.getAsJsonArray("models").forEach(el -> {
            JsonObject m = el.getAsJsonObject();
            String name = m.get("name").getAsString();
            boolean available = m.get("available").getAsBoolean();
            boolean isCurrent = name.equals(current);
            Formatting color = !available ? Formatting.DARK_GRAY
                    : isCurrent ? Formatting.GREEN : Formatting.WHITE;
            String flags = (m.get("supports_text").getAsBoolean() ? " [text]" : "")
                    + (m.get("loaded").getAsBoolean() ? " [loaded]" : "")
                    + (available ? "" : " [unavailable: "
                        + m.get("unavailable_reason").getAsString() + "]");
            src.sendFeedback(() -> Text.literal((isCurrent ? "> " : "  ") + name + flags)
                    .formatted(color), false);
            src.sendFeedback(() -> Text.literal("    " + m.get("description").getAsString())
                    .formatted(Formatting.GRAY), false);
        });
        src.sendFeedback(() -> Text.literal("/model <name> to switch").formatted(Formatting.GRAY),
                false);
    }

    private static int setModel(CommandContext<ServerCommandSource> ctx) {
        String name = StringArgumentType.getString(ctx, "name");
        BlockGenConfig cfg = BlockGenConfig.get();
        cfg.model = "default".equals(name) ? null : name;
        cfg.save();
        ctx.getSource().sendFeedback(() -> Text.literal("Model set to " + name
                + " (not verified until the next /gen)").formatted(Formatting.GREEN), false);
        return 1;
    }

    private static int showServer(CommandContext<ServerCommandSource> ctx) {
        ctx.getSource().sendFeedback(
                () -> Text.literal("Server: " + BlockGenConfig.get().serverUrl), false);
        return 1;
    }

    private static int setServer(CommandContext<ServerCommandSource> ctx) {
        String url = StringArgumentType.getString(ctx, "url").trim();
        if (!url.startsWith("ws://") && !url.startsWith("wss://")) {
            ctx.getSource().sendError(Text.literal(
                    "URL must start with ws:// or wss:// (e.g. ws://192.168.1.20:8000/ws)"));
            return 0;
        }
        BlockGenConfig cfg = BlockGenConfig.get();
        cfg.serverUrl = url;
        cfg.save();
        ctx.getSource().sendFeedback(
                () -> Text.literal("Server set to " + url).formatted(Formatting.GREEN), false);
        return 1;
    }

    private static int setSpeed(CommandContext<ServerCommandSource> ctx) {
        int n = com.mojang.brigadier.arguments.IntegerArgumentType.getInteger(ctx, "n");
        BlockGenConfig cfg = BlockGenConfig.get();
        cfg.blocksPerTick = n;
        cfg.save();
        ctx.getSource().sendFeedback(() -> Text.literal(
                "Placing " + n + " blocks/tick (~" + (n * 20) + "/s)").formatted(Formatting.GREEN),
                false);
        return 1;
    }

    /** Reports connection/protocol errors from a bare model listing back to the caller. */
    private record ReportingHandler(ServerCommandSource src) implements InferenceClient.Handler {
        @Override public void onBegin(String model, long seed, String prompt) {}
        @Override public void onBlocks(java.util.List<InferenceClient.PendingBlock> blocks) {}
        @Override public void onDone(int blocks, double elapsed, String reason) {}
        @Override public void onError(String message) {
            src.getServer().execute(() ->
                    src.sendError(Text.literal("BlockGen: " + message)));
        }
    }
}

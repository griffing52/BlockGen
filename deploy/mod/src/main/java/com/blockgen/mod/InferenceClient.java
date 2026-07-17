package com.blockgen.mod;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.WebSocket;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.atomic.AtomicReference;

/**
 * WebSocket client for the BlockGen inference server.
 *
 * <p>Uses the JDK's own {@link WebSocket} so the mod needs no third-party
 * dependency. Two things about that API are easy to get wrong and both are handled
 * here:
 *
 * <ul>
 *   <li><b>Frames are not messages.</b> {@code onText} can deliver a message in
 *       pieces, so text is accumulated until {@code last} is true. A batch of blocks
 *       is comfortably large enough to be split, so parsing each fragment as JSON
 *       would fail intermittently under exactly the conditions we care about.
 *   <li><b>Callbacks run on HttpClient threads.</b> Nothing here touches the world;
 *       it hands parsed messages to the {@link Handler}, and {@link GenerationSession}
 *       queues them for the server thread. Touching world state from this thread is
 *       a race against the tick loop.
 * </ul>
 *
 * <p>Every connection is single-use: one connect, one generation, one close. That
 * keeps session state trivially correct at the cost of a handshake per build (~ms on
 * a LAN, against a ~10s generation).
 */
public class InferenceClient {

    /** Callbacks are invoked on WebSocket threads, never the server thread. */
    public interface Handler {
        void onBegin(String model, long seed, String prompt);
        void onBlocks(List<PendingBlock> blocks);
        void onDone(int blocks, double elapsed, String reason);
        void onError(String message);
    }

    /** One block placement in the model's local frame, before anchoring. */
    public record PendingBlock(int x, int y, int z, String state) {}

    private final HttpClient http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .build();
    private final AtomicReference<WebSocket> socket = new AtomicReference<>();

    /** Opens a connection and starts a generation; failures surface via the handler. */
    public void generate(String url, String model, String prompt, Long seed,
                         BlockGenConfig cfg, Handler handler) {
        JsonObject req = new JsonObject();
        req.addProperty("type", "generate");
        if (model != null) req.addProperty("model", model);
        if (prompt != null) req.addProperty("prompt", prompt);
        if (seed != null) req.addProperty("seed", seed);
        req.addProperty("temperature", cfg.temperature);
        req.addProperty("top_k", cfg.topK);
        req.addProperty("cfg_scale", cfg.cfgScale);
        send(url, req, handler);
    }

    /** Asks the server for its model list. Reuses the same message pump. */
    public void listModels(String url, Handler handler, java.util.function.Consumer<JsonObject> onModels) {
        JsonObject req = new JsonObject();
        req.addProperty("type", "models");
        connect(url, handler, onModels, req);
    }

    private void send(String url, JsonObject req, Handler handler) {
        connect(url, handler, null, req);
    }

    private void connect(String url, Handler handler,
                         java.util.function.Consumer<JsonObject> onModels, JsonObject req) {
        URI uri;
        try {
            uri = URI.create(url);
        } catch (IllegalArgumentException e) {
            handler.onError("Bad server URL '" + url + "': " + e.getMessage());
            return;
        }
        http.newWebSocketBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .buildAsync(uri, new Listener(handler, onModels))
                .whenComplete((ws, err) -> {
                    if (err != null) {
                        handler.onError("Could not reach " + url + " -- is the inference "
                                + "server running? (" + rootCause(err) + ")");
                        return;
                    }
                    socket.set(ws);
                    ws.sendText(req.toString(), true);
                });
    }

    /** Asks the server to stop the current generation. Safe to call when idle. */
    public void cancel() {
        WebSocket ws = socket.get();
        if (ws != null && !ws.isOutputClosed()) {
            JsonObject req = new JsonObject();
            req.addProperty("type", "cancel");
            ws.sendText(req.toString(), true);
        }
    }

    public void close() {
        WebSocket ws = socket.getAndSet(null);
        if (ws != null && !ws.isOutputClosed()) {
            ws.sendClose(WebSocket.NORMAL_CLOSURE, "done");
        }
    }

    private static String rootCause(Throwable t) {
        Throwable c = t;
        while (c.getCause() != null) c = c.getCause();
        return c.getClass().getSimpleName() + ": " + c.getMessage();
    }

    private final class Listener implements WebSocket.Listener {
        private final Handler handler;
        private final java.util.function.Consumer<JsonObject> onModels;
        private final StringBuilder buffer = new StringBuilder();

        Listener(Handler handler, java.util.function.Consumer<JsonObject> onModels) {
            this.handler = handler;
            this.onModels = onModels;
        }

        @Override
        public void onOpen(WebSocket webSocket) {
            webSocket.request(1);
        }

        @Override
        public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
            buffer.append(data);
            if (last) {
                String json = buffer.toString();
                buffer.setLength(0);
                try {
                    dispatch(JsonParser.parseString(json).getAsJsonObject());
                } catch (RuntimeException e) {
                    handler.onError("Malformed message from server: " + e);
                }
            }
            webSocket.request(1);
            return null;
        }

        private void dispatch(JsonObject msg) {
            String type = msg.has("type") ? msg.get("type").getAsString() : "";
            switch (type) {
                case "begin" -> handler.onBegin(
                        msg.get("model").getAsString(),
                        msg.get("seed").getAsLong(),
                        msg.has("prompt") && !msg.get("prompt").isJsonNull()
                                ? msg.get("prompt").getAsString() : null);
                case "blocks" -> {
                    JsonArray arr = msg.getAsJsonArray("blocks");
                    List<PendingBlock> out = new java.util.ArrayList<>(arr.size());
                    for (JsonElement e : arr) {
                        JsonArray b = e.getAsJsonArray();
                        out.add(new PendingBlock(b.get(0).getAsInt(), b.get(1).getAsInt(),
                                b.get(2).getAsInt(), b.get(3).getAsString()));
                    }
                    handler.onBlocks(out);
                }
                case "done" -> {
                    handler.onDone(msg.get("blocks").getAsInt(),
                            msg.get("elapsed").getAsDouble(),
                            msg.get("reason").getAsString());
                    close();
                }
                case "models" -> {
                    if (onModels != null) onModels.accept(msg);
                    close();
                }
                case "error" -> {
                    handler.onError(msg.get("message").getAsString());
                    close();
                }
                default -> handler.onError("Unknown message type from server: " + type);
            }
        }

        @Override
        public CompletionStage<?> onClose(WebSocket webSocket, int statusCode, String reason) {
            socket.set(null);
            return null;
        }

        @Override
        public void onError(WebSocket webSocket, Throwable error) {
            socket.set(null);
            handler.onError("Connection error: " + rootCause(error));
        }
    }
}

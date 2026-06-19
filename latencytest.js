import fs from "fs";
import path from "path";
import dotenv from "dotenv";
dotenv.config({ quiet: true })

const BASE_URL = "http://localhost:20128/api/v1";
const API_KEY = process.env.LOCAL_API_KEY ?? "";
const MODEL_INPUT = "GLM 5.2"; // plain string — auto-converted to a precise regex below

// Builds a regex from a human string: escapes dots, treats space/dash as equivalent,
// and blocks partial suffix matches (e.g. "GLM-5.2-FP8").
function buildFilter(str) {
    // Normalize: replace spaces and dashes in the input with a placeholder, then build pattern
    const normalized = str.trim().replace(/[\s-]+/g, " "); // collapse spaces/dashes
    const parts = normalized.split(" ").map(p => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")); // escape each word
    const pattern = parts.join("[\\s-]") + "(?:-fp8)?(?![\\w-])";   // join with space-or-dash, block suffix except optional -fp8
    return new RegExp(pattern, "i");
}

const MODEL_FILTER = buildFilter(MODEL_INPUT);
const PROMPT = "51324214123 + 3213131232 (return just the number, no explanation)";
const EXPECTED_ANSWER = "54537345355";
const TRIES = 2;
const RESULTS_DIR = path.join("latency_results");

const headers = {
    authorization: `Bearer ${API_KEY}`,
    "content-type": "application/json",
};

async function getModels() {
    const res = await fetch(`${BASE_URL}/models`, { headers });
    const json = await res.json();
    fs.writeFileSync("local_models.json", JSON.stringify(json, null, 2));

    // Automatically output the list of vision: true models
    const visionModels = (json.data ?? json.models ?? [])
        .filter(m => m.capabilities?.vision === true)
        .map(m => m.id ?? m.name);
    fs.writeFileSync("vision_models.txt", visionModels.join("\n"));

    // OpenAI-compatible: { data: [ { id, ... }, ... ] }
    return (json.data ?? json.models ?? []).map((m) => m.id ?? m.name ?? m);
}

async function ask(model) {
    const start = Date.now();
    const res = await fetch(`${BASE_URL}/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify({
            model,
            stream: true,
            stream_options: { include_usage: true }, // ask server to include usage in final chunk
            messages: [{ role: "user", content: PROMPT }],
        }),
    });

    // --- Parse the SSE stream ---
    let ttftMs = null;          // time to first token
    let answer = "";
    let chunkCount = 0;         // fallback completion-token counter
    let usage = null;

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    outer: while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // keep incomplete last line

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data:")) continue;
            const data = trimmed.slice(5).trim();
            if (data === "[DONE]") break outer;

            let chunk;
            try { chunk = JSON.parse(data); } catch { continue; }

            // Capture usage from the final chunk (stream_options: include_usage)
            if (chunk.usage) usage = chunk.usage;

            const delta = chunk.choices?.[0]?.delta?.content;
            if (delta) {
                if (ttftMs === null) ttftMs = Date.now() - start; // first token!
                answer += delta;
                chunkCount++;
            }
        }
    }

    const latencyMs = Date.now() - start;
    answer = answer.trim();

    // Extract token counts — prefer server-reported usage, fall back to chunk count
    const promptTokens = usage?.prompt_tokens ?? usage?.input_tokens ?? null;
    const completionTokens = usage?.completion_tokens ?? usage?.output_tokens ?? chunkCount;
    const totalTokens = usage?.total_tokens ?? null;

    // Tokens per second based on total latency
    const tokensPerSec =
        completionTokens != null && latencyMs > 0
            ? (completionTokens / (latencyMs / 1000)).toFixed(2)
            : null;

    return { answer, latencyMs, ttftMs, promptTokens, completionTokens, totalTokens, tokensPerSec };
}

async function main() {
    fs.mkdirSync(RESULTS_DIR, { recursive: true });

    const allModels = await getModels();
    const matched = allModels.filter((id) => MODEL_FILTER.test(id));

    if (matched.length === 0) {
        console.log("No models matched filter:", MODEL_FILTER);
        return;
    }

    console.log(`Found ${matched.length} matching model(s):`, matched);

    let csvContent = "Model,Try,Latency_ms,TTFT_ms,PromptTokens,CompletionTokens,TotalTokens,Tokens_per_sec,Answer\n";

    for (const model of matched) {
        for (let i = 1; i <= TRIES; i++) {
            const { answer, latencyMs, ttftMs, promptTokens, completionTokens, totalTokens, tokensPerSec } = await ask(model);
            const correct = answer === EXPECTED_ANSWER;

            const tpsLabel = tokensPerSec != null ? `${tokensPerSec} tok/s` : "n/a tok/s";
            const ttftLabel = ttftMs != null ? `TTFT ${ttftMs}ms` : "TTFT n/a";
            const tokenLabel = completionTokens != null
                ? `(${promptTokens ?? "?"} prompt + ${completionTokens} completion = ${totalTokens ?? "?"} total)`
                : "(tokens: n/a)";

            if (correct) {
                console.log(`[${model}] Try ${i}: ✅ "${answer}" — total ${latencyMs}ms | ${ttftLabel} | ${tpsLabel} ${tokenLabel}`);
                csvContent += `"${model}",${i},${latencyMs},${ttftMs ?? ""},${promptTokens ?? ""},${completionTokens ?? ""},${totalTokens ?? ""},${tokensPerSec ?? ""},"${answer.replace(/"/g, '""')}"\n`;
            } else {
                console.log(`[${model}] Try ${i}: ❌ "${answer}" (skipped, not ${EXPECTED_ANSWER}) — total ${latencyMs}ms | ${ttftLabel} | ${tpsLabel} ${tokenLabel}`);
            }
        }
    }

    const outFile = path.join(
        RESULTS_DIR,
        `latency_${new Date().toISOString().replace(/[:.]/g, "-")}.csv`
    );
    fs.writeFileSync(outFile, csvContent);
    console.log(`\nResults saved to ${outFile}`);
}

main().catch(console.error);

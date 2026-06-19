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
const PROMPT = "2+2 (return just the number, no explanation)";
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
            stream: false,
            messages: [{ role: "user", content: PROMPT }],
        }),
    });
    const latency = Date.now() - start;
    const json = await res.json();
    const answer = json.choices?.[0]?.message?.content?.trim() ?? "";
    return { answer, latency };
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

    let csvContent = "Model,Try,Latency,Answer\n";

    for (const model of matched) {
        for (let i = 1; i <= TRIES; i++) {
            const { answer, latency } = await ask(model);
            const correct = answer === "4";
            if (correct) {
                console.log(`[${model}] Try ${i}: ✅ "${answer}" — ${latency}ms`);
                csvContent += `"${model}",${i},${latency},"${answer.replace(/"/g, '""')}"\n`;
            } else {
                console.log(`[${model}] Try ${i}: ❌ "${answer}" (skipped, not 4)`);
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

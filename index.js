import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// ─── Config ──────────────────────────────────────────────────────────────────

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ATTEMPTS = 2;
const PROMPT = 'Return only the names of the players in the lobby/party shown in this image, one per line. No extra text. Do not repeat names';

const PROVIDERS = [
    {
        name: 'OpenRouter',
        models: [
            'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
            'nvidia/nemotron-nano-12b-v2-vl:free',
            'nex-agi/nex-n2-pro:free',
        ],
        call: (model, test) => callOpenRouter(model, test),
    },
    {
        name: 'Google AI Studio',
        models: [
            'gemma-4-26b-a4b-it'
        ],
        call: (model, test) => callGoogleAI(model, test),
    },
    {
        name: 'Cloudflare',
        models: [
            '@cf/mistralai/mistral-small-3.1-24b-instruct', // vision
            '@cf/meta/llama-4-scout-17b-16e-instruct', // vision
            // '@cf/google/gemma-4-26b-a4b-it',              // vision
        ],
        call: (model, test) => callCloudflare(model, test),
    },
    {
        name: 'Groq (base64)',
        models: [
            'meta-llama/llama-4-scout-17b-16e-instruct',
        ],
        call: (model, test) => callGroq(model, test.imageDataUrl),
    },
    {
        name: 'Groq (url)',
        models: [
            'meta-llama/llama-4-scout-17b-16e-instruct',
        ],
        call: (model, test) => callGroq(model, test.imageUrl),
    },
    {
        name: 'NVIDIA (base64)',
        models: [
            'moonshotai/kimi-k2.6',
            'meta/llama-4-maverick-17b-128e-instruct',
        ],
        call: (model, test) => callNvidia(model, test.imageDataUrl),
    },
    {
        name: 'NVIDIA (url)',
        models: [
            'moonshotai/kimi-k2.6',
            'meta/llama-4-maverick-17b-128e-instruct',
        ],
        call: (model, test) => callNvidia(model, test.imageUrl),
    },
    {
        name: 'Ollama Cloud',
        models: [
            'gemma4:31b-cloud',
            'minimax-m3:cloud',
            'ministral-3:3b-cloud',
            'ministral-3:8b-cloud',
            'ministral-3:14b-cloud',
        ],
        call: (model, test) => callOllamaCloud(model, test),
    },
];

// ─── Load Tests ───────────────────────────────────────────────────────────────
// Each test is a pair of files in image_tests/:
//   <name>.png  (or .jpg / .jpeg / .webp)  ← image
//   <name>.txt                              ← one ground-truth name per line
//
// Optional image URLs are read from .env as IMAGE_URL_1, IMAGE_URL_2, …
// matched by the alphabetical order of the discovered image files.

const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.webp']);
const MIME_MAP = { '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp' };

function loadTests() {
    const testsDir = path.join(__dirname, 'image_tests');
    const files = fs.readdirSync(testsDir).sort();

    const imageFiles = files.filter(f => IMAGE_EXTS.has(path.extname(f).toLowerCase()));

    if (imageFiles.length === 0) {
        throw new Error('No image files found in image_tests/. Add at least one image and its matching .txt ground truth.');
    }

    return imageFiles.map((imgFile) => {
        const stem = path.basename(imgFile, path.extname(imgFile));
        const ext = path.extname(imgFile).toLowerCase();
        const mime = MIME_MAP[ext] ?? 'image/png';
        const imgPath = path.join(testsDir, imgFile);
        const txtPath = path.join(testsDir, `${stem}.txt`);

        if (!fs.existsSync(txtPath)) {
            throw new Error(`Missing ground truth file: image_tests/${stem}.txt`);
        }

        const imageBase64 = fs.readFileSync(imgPath).toString('base64');
        const imageDataUrl = `data:${mime};base64,${imageBase64}`;
        const envKey = `IMAGE_URL_${stem.toUpperCase()}`;
        const imageUrl = process.env[envKey] ?? null;

        const groundTruth = fs.readFileSync(txtPath, 'utf8')
            .split(/\r?\n/)
            .map(l => l.trim())
            .filter(Boolean)
            .map(n => n.toLowerCase());

        return { name: stem, mime, imageBase64, imageDataUrl, imageUrl, groundTruth };
    });
}

const TESTS = loadTests();

// Warn about missing image URLs
TESTS.forEach((test) => {
    if (!test.imageUrl) {
        const envKey = `IMAGE_URL_${test.name.toUpperCase()}`;
        console.warn(`⚠  ${envKey} not set in .env — URL-based providers will skip test "${test.name}".`);
    }
});

// ─── Helpers ─────────────────────────────────────────────────────────────────

const TIMEOUT_MS = 40_000;

function fetchWithTimeout(url, options) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    return fetch(url, { ...options, signal: controller.signal })
        .finally(() => clearTimeout(timer));
}

// ─── Provider Adapters ────────────────────────────────────────────────────────

async function callOpenRouter(model, test) {
    const t0 = Date.now();
    const res = await fetchWithTimeout('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model,
            messages: [{
                role: 'user',
                content: [
                    { type: 'text', text: PROMPT },
                    { type: 'image_url', image_url: { url: test.imageDataUrl } },
                ],
            }],
            reasoning: { enabled: false },
        }),
    });
    const responseTimeMs = Date.now() - t0;
    const data = await res.json();
    const apiError = res.ok ? null : `HTTP ${res.status}: ${data?.error?.message ?? JSON.stringify(data?.error ?? data)}`;
    const text = data?.choices?.[0]?.message?.content ?? '';
    return { text, responseTimeMs, apiError };
}

async function callGoogleAI(model, test) {
    const t0 = Date.now();
    const res = await fetchWithTimeout(
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${process.env.GOOGLE_AI_STUDIO_KEY}`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                contents: [{
                    parts: [
                        { inlineData: { mimeType: test.mime, data: test.imageBase64 } },
                        { text: PROMPT },
                    ],
                }],
            }),
        }
    );
    const responseTimeMs = Date.now() - t0;
    const data = await res.json();
    const apiError = res.ok ? null : `HTTP ${res.status}: ${data?.error?.message ?? JSON.stringify(data?.error ?? data)}`;
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
    return { text, responseTimeMs, apiError };
}

// Track which CF models have already agreed to their license
const cfAgreedModels = new Set();

async function cfRequest(model, body) {
    const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
    return fetchWithTimeout(
        `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/run/${model}`,
        {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${process.env.CLOUDFLARE_API_KEY}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        }
    );
}

async function callCloudflare(model, test) {
    const t0 = Date.now();

    // Auto-agree to model license on first use
    if (!cfAgreedModels.has(model)) {
        const agreeRes = await cfRequest(model, { messages: [{ role: 'user', content: 'agree' }] });
        if (agreeRes.ok || agreeRes.status === 400) {
            // 400 means model got the message but rejected content format — that's fine, still agreed
            cfAgreedModels.add(model);
        } else if (agreeRes.status !== 403) {
            cfAgreedModels.add(model); // Not a 403, no agreement needed
        } else {
            const body = await agreeRes.json();
            if (!body?.errors?.[0]?.message?.includes('Model Agreement')) {
                cfAgreedModels.add(model);
            }
            // else: still 403 with agreement error — proceed anyway and let main request fail
        }
    }

    const res = await cfRequest(model, {
        messages: [{
            role: 'user',
            content: [
                { type: 'image_url', image_url: { url: test.imageDataUrl } },
                { type: 'text', text: PROMPT },
            ],
        }],
    });

    const responseTimeMs = Date.now() - t0;
    const data = await res.json();
    const apiError = res.ok ? null : `HTTP ${res.status}: ${data?.errors?.[0]?.message ?? JSON.stringify(data?.errors ?? data)}`;
    // Cloudflare returns result.response for most models; gemma/others may use result.choices
    const text = data?.result?.response
        ?? data?.result?.choices?.[0]?.message?.content
        ?? '';
    return { text, responseTimeMs, apiError };
}

async function callGroq(model, imageSource) {
    if (!imageSource) {
        return { text: '', responseTimeMs: 0, apiError: 'No image source available for this test' };
    }
    const t0 = Date.now();
    const res = await fetchWithTimeout('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model,
            messages: [{
                role: 'user',
                content: [
                    { type: 'text', text: PROMPT },
                    { type: 'image_url', image_url: { url: imageSource } },
                ],
            }],
        }),
    });
    const responseTimeMs = Date.now() - t0;
    const data = await res.json();
    const apiError = res.ok ? null : `HTTP ${res.status}: ${data?.error?.message ?? JSON.stringify(data?.error ?? data)}`;
    const text = data?.choices?.[0]?.message?.content ?? '';
    return { text, responseTimeMs, apiError };
}

async function callNvidia(model, imageSource) {
    if (!imageSource) {
        return { text: '', responseTimeMs: 0, apiError: 'No image source available for this test' };
    }
    const t0 = Date.now();
    // NVIDIA supports image_url objects (URL or base64 data URL).
    // For images >180KB via base64, NVCF Asset API upload is recommended.
    // URL mode avoids the size limit entirely.
    const res = await fetchWithTimeout('https://integrate.api.nvidia.com/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${process.env.NVIDIA_API_KEY}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model,
            messages: [{
                role: 'user',
                content: [
                    { type: 'text', text: PROMPT },
                    { type: 'image_url', image_url: { url: imageSource } },
                ],
            }],
            max_tokens: 512,
            temperature: 0.2,
            top_p: 0.7,
            stream: false,
        }),
    });
    const responseTimeMs = Date.now() - t0;
    const data = await res.json();
    const apiError = res.ok ? null : `HTTP ${res.status}: ${data?.detail ?? data?.error?.message ?? JSON.stringify(data?.error ?? data)}`;
    const text = data?.choices?.[0]?.message?.content ?? '';
    return { text, responseTimeMs, apiError };
}

async function callOllamaCloud(model, test) {
    const t0 = Date.now();
    const res = await fetchWithTimeout('https://ollama.com/api/chat', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${process.env.OLLAMA_API_KEY}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model,
            messages: [{
                role: 'user',
                content: PROMPT,
                images: [test.imageBase64],
            }],
            stream: false,
        }),
    });
    const responseTimeMs = Date.now() - t0;
    const data = await res.json();
    const apiError = res.ok ? null : `HTTP ${res.status}: ${data?.error ?? JSON.stringify(data)}`;
    const text = data?.message?.content ?? '';
    return { text, responseTimeMs, apiError };
}

// ─── Accuracy ─────────────────────────────────────────────────────────────────

function stripThinking(text) {
    // Remove <think>...</think> blocks (including nested/malformed)
    return text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
}

function parseNames(text) {
    return stripThinking(text)
        .split(/\r?\n/)
        .map(l => l.trim().replace(/^[-*•\d.]+\s*/, ''))
        .filter(Boolean)
        .map(n => n.toLowerCase());
}

function calcAccuracy(names, groundTruth) {
    // Each ground truth name can only be matched once (prevents >100%)
    const remaining = [...groundTruth];
    for (const n of names) {
        const idx = remaining.findIndex(gt => gt === n || n.includes(gt) || gt.includes(n));
        if (idx !== -1) remaining.splice(idx, 1);
    }
    const matchedCount = groundTruth.length - remaining.length;
    return ((matchedCount / groundTruth.length) * 100).toFixed(1);
}

// ─── CSV ──────────────────────────────────────────────────────────────────────

const resultsDir = path.join(__dirname, 'results');
if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir);

const timestamp = new Date().toISOString().replace(/[:.]/g, '-').replace('T', '_').slice(0, 19);
const csvPath = path.join(resultsDir, `results_${timestamp}.csv`);
const csvHeader = 'provider,model,test,attempt,names_found,accuracy_pct,response_time_ms,error\n';
fs.writeFileSync(csvPath, csvHeader);

function appendCsv(row) {
    fs.appendFileSync(csvPath, row + '\n');
}

// ─── Runner ───────────────────────────────────────────────────────────────────

async function runModel(provider, model, test) {
    console.log(`\n▶ ${provider.name} | ${model} | [${test.name}]`);

    const tasks = Array.from({ length: ATTEMPTS }, (_, i) =>
        provider.call(model, test).then(r => ({ attempt: i + 1, ...r })).catch(err => ({
            attempt: i + 1,
            text: '',
            responseTimeMs: 0,
            fetchError: err.name === 'AbortError' ? 'TIMEOUT (>40s)' : err.message,
        }))
    );

    const results = await Promise.all(tasks);

    for (const { attempt, text, responseTimeMs, apiError, fetchError } of results) {
        const errorMsg = fetchError ?? apiError ?? '';

        if (fetchError) {
            console.log(`  Attempt ${attempt}: ERROR — ${fetchError}`);
            appendCsv(`"${provider.name}","${model}","${test.name}",${attempt},0,0,0,"${fetchError}"`);
            continue;
        }

        const names = parseNames(text);
        const accuracy = calcAccuracy(names, test.groundTruth);

        if (apiError) {
            console.log(`  Attempt ${attempt}: API ERROR — ${apiError}`);
        } else {
            console.log(`  Attempt ${attempt}: ${accuracy}% accuracy | ${responseTimeMs}ms | found: [${names.join(', ')}]`);
        }

        appendCsv(`"${provider.name}","${model}","${test.name}",${attempt},${names.length},${accuracy},${responseTimeMs},"${errorMsg.replace(/"/g, "'")}"`);
    }
}

async function main() {
    console.log('═══════════════════════════════════════════');
    console.log('  AI Model Benchmark — Image Name Recognition');
    console.log(`  Tests: ${TESTS.map(t => t.name).join(', ')}`);
    console.log('═══════════════════════════════════════════');

    for (const provider of PROVIDERS) {
        for (const model of provider.models) {
            for (const test of TESTS) {
                await runModel(provider, model, test);
            }
        }
    }

    console.log(`\n✓ Done. Results saved to ${csvPath}`);
}

main().catch(console.error);

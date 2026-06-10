# AI Vision Model Benchmark

This repository contains a Node.js benchmark tool designed to evaluate and compare the OCR (Optical Character Recognition) and text-extraction capabilities, speed, and reliability of various multimodal AI models across multiple API providers.

Specifically, it benchmarks the models' ability to extract player names from a League of Legends ARAM game lobby screenshot and compares the results against a predefined ground truth list.

---

## What It Is Used For

The tool is used to:
1. **Measure Accuracy:** Compare the list of names extracted by each model against the ground truth list of 10 names in [response.txt](file:///c:/Users/user/OneDrive/Documents/Projects/BenchmarkModels/response.txt) (order-insensitive).
2. **Measure Performance:** Track the request response time in milliseconds for each attempt.
3. **Assess Consistency:** Run **3 concurrent requests** per model to calculate average success rates and performance variance.
4. **Compare Input Methods:** Compare performance differences between passing images as **Base64 data URLs** vs. passing direct **Image URLs** (supported by Groq and NVIDIA).
5. **Log Errors:** Capture API error codes, response status issues, and network timeouts (configured at 40 seconds).
6. **Generate Reports:** Automatically save the detailed results of all attempts into a timestamped CSV report within the `results/` directory.

---

## Features

- **Multi-Provider Support:** Adapters for OpenRouter, Google AI Studio, Cloudflare Workers AI, Groq, and NVIDIA.
- **Concurrent Execution:** Sends 3 requests simultaneously per model to get accurate average response times and verify model temperature consistency.
- **Thinking Tags Extraction:** Automatically strips `<think>...</think>` tags from reasoning models (e.g., DeepSeek R1 variants) to parse name listings cleanly.
- **Fuzzy Matching:** Performs standard and substring matching to prevent false negatives caused by minor capitalization or formatting issues.

---

## Setup & Configuration

### 1. Installation
Clone the repository and install dependencies:
```bash
npm install
```

### 2. Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
# API Keys (Provide keys for the models you want to test)
OPENROUTER_API_KEY=your_openrouter_api_key
GOOGLE_AI_STUDIO_KEY=your_google_ai_studio_key
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
CLOUDFLARE_API_KEY=your_cloudflare_api_key
GROQ_API_KEY=your_groq_api_key
NVIDIA_API_KEY=your_nvidia_api_key

# Public URL of the image for URL-based benchmarking (Groq and NVIDIA)
IMAGE_URL=https://example.com/path/to/your/aram.png
```

### 3. Input Files
- **Test Image:** Place the screenshot to be processed as `aram.png` in the project root directory.
- **Ground Truth:** Add the expected player names (one per line) in [response.txt](file:///c:/Users/user/OneDrive/Documents/Projects/BenchmarkModels/response.txt).

---

## Usage

Run the benchmark script [index.js](file:///c:/Users/user/OneDrive/Documents/Projects/BenchmarkModels/index.js):
```bash
node index.js
```


### Console Output Example
```text
═══════════════════════════════════════════
  AI Model Benchmark — Image Name Recognition
  Ground truth: 10 names
═══════════════════════════════════════════

▶ Groq (url) | meta-llama/llama-4-scout-17b-16e-instruct
  Attempt 1: 100.0% accuracy | 755ms | found: [shank god, chasing silver, crazykoala, ganzikus, apple watch, dang, fizzzl, ericx12138, imbaqinz, frost sab
re]
  Attempt 2: 100.0% accuracy | 677ms | found: [shank god, chasing silver, crazykoala, ganzikus, apple watch, dang, fizzzl, ericx12138, imbaqinz, frost sab
re]
  Attempt 3: 100.0% accuracy | 627ms | found: [shank god, chasing silver, crazykoala, ganzikus, apple watch, dang, fizzzl, ericx12138, imbaqinz, frost sab
re]

✓ Done. Results saved to results/results_2026-06-10_02-30-00.csv
```

### CSV Output Format
The resulting CSV file inside the `results/` folder contains the following fields:
`provider,model,attempt,names_found,accuracy_pct,response_time_ms,error`
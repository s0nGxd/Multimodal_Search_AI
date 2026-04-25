Instructions: Calling Gemini Flash via Vertex AI
You are adding a Gemini Flash call to a JavaScript project using the Vercel AI SDK. Follow this template exactly.

1. Credentials
Credentials live in /server.env. Load them before instantiating the client. The three required variables are:


GOOGLE_PROJECT_ID=...
GOOGLE_CLIENT_EMAIL=...
GOOGLE_PRIVATE_KEY=...
GOOGLE_PRIVATE_KEY is usually stored with literal \n sequences and often wrapped in quotes — normalize it before passing it to the SDK (see formatPrivateKey below).

2. Install dependencies

pnpm add ai @ai-sdk/google-vertex zod
(Use npm or yarn if that's the project's package manager. zod is only needed for structured output.)

3. Create the Vertex client
Create a single shared client module. Do not instantiate a new client per call site.


// lib/vertex.js
import { createVertex } from "@ai-sdk/google-vertex";

function formatPrivateKey(key) {
  if (!key) return undefined;
  return key
    .replace(/^"(.*)"$/, "$1") // strip surrounding quotes
    .replace(/\\n/g, "\n")      // convert literal \n to real newlines
    .trim();
}

const PROJECT_ID = process.env.GOOGLE_PROJECT_ID;

// Force the global publisher path. The SDK's auto-constructed global URL
// (https://global-aiplatform.googleapis.com/...) returns 404/429 even when
// capacity exists. This baseURL routes quota to the correct pool.
export const vertex = createVertex({
  project: PROJECT_ID,
  location: "global",
  baseURL: `https://aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/global/publishers/google`,
  googleAuthOptions: {
    credentials: {
      client_email: process.env.GOOGLE_CLIENT_EMAIL,
      private_key: formatPrivateKey(process.env.GOOGLE_PRIVATE_KEY),
    },
  },
});
4. Make the call
Structured output (recommended for classification, extraction, gates)

import { generateObject } from "ai";
import { z } from "zod";
import { vertex } from "./lib/vertex.js";

const MODEL = "gemini-2.5-flash";

const schema = z.object({
  category: z.string().describe("The post category"),
  confidence: z.number().describe("0-1 confidence score"),
});

const result = await generateObject({
  model: vertex(MODEL),
  schema,
  prompt: "...",
});

const { category, confidence } = result.object;
Add .describe(...) on every Zod field — the model reads those as per-field instructions.

Freeform text

import { generateText } from "ai";
import { vertex } from "./lib/vertex.js";

const result = await generateText({
  model: vertex("gemini-2.5-flash"),
  prompt: "...",
});

const text = result.text;
5. Handle 429 quota errors
Vertex returns 429s under load. Wrap every call in a retry helper with exponential backoff. Minimal version:


async function withQuotaRetry(fn, { label, maxAttempts = 5 } = {}) {
  let delay = 1000;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const is429 = err?.statusCode === 429 || /429|quota|rate/i.test(err?.message ?? "");
      if (!is429 || attempt === maxAttempts) throw err;
      console.warn(`[${label}] 429 attempt ${attempt}, retrying in ${delay}ms`);
      await new Promise((r) => setTimeout(r, delay));
      delay *= 2;
    }
  }
}

// Usage
const result = await withQuotaRetry(
  () => generateObject({ model: vertex(MODEL), schema, prompt: "..." }),
  { label: "classify-post" }
);
Always pass a label — it's how you'll identify which call site is being rate-limited.



---------

to pass images:

Swap prompt: "..." for a messages array with a multimodal user turn. The model stays the same — gemini-2.5-flash handles vision natively.


import { generateObject } from "ai";
import { z } from "zod";
import { vertex } from "./lib/vertex.js";
import fs from "node:fs";

const schema = z.object({
  matches: z.boolean().describe("Does the image match the query?"),
  reasoning: z.string(),
});

const result = await generateObject({
  model: vertex("gemini-2.5-flash"),
  schema,
  messages: [
    {
      role: "user",
      content: [
        { type: "text", text: "Does this image match: 'a red bicycle in a city'?" },
        {
          type: "image",
          image: fs.readFileSync("./bike.jpg"), // Buffer, Uint8Array, URL, or base64 string
          mediaType: "image/jpeg",
        },
      ],
    },
  ],
});
What changes vs. text-only:

prompt → messages (required for multimodal).
content is an array of parts, not a string.
Each image is { type: "image", image: <data>, mediaType: "image/..." }.
image accepts: Buffer, Uint8Array, base64 string, URL, or a remote URL string.
Put the text part and image part(s) in the same content array — order matters; the model reads them in sequence.
Multiple images: just add more { type: "image", ... } entries.
Everything else (client, retry wrapper, schema, model ID) stays identical.

________
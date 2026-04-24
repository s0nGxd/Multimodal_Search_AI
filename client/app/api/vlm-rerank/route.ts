import { NextRequest, NextResponse } from "next/server";
import { generateObject } from "ai";
import { z } from "zod";
import { vertex, withQuotaRetry } from "@/lib/vertex";
import sharp from "sharp";

export const runtime = "nodejs";
export const maxDuration = 30;

async function fetchAndResize(url: string): Promise<Buffer | null> {
    try {
        const res = await fetch(url, { signal: AbortSignal.timeout(6000) });
        if (!res.ok) return null;
        const raw = Buffer.from(await res.arrayBuffer());
        // Resize to max 768px, JPEG q=75 — brings 9MB photos to ~80-150KB,
        // which Vertex ingests reliably and Gemini still sees clearly.
        return await sharp(raw).resize({ width: 768, withoutEnlargement: true }).jpeg({ quality: 75 }).toBuffer();
    } catch {
        return null;
    }
}

const ItemSchema = z.object({
    photo_id: z.string(),
    photo_image_url: z.string(),
});

const BodySchema = z.object({
    query: z.string().min(1),
    candidates: z.array(ItemSchema).min(1).max(12),
});

const VlmScore = z.object({
    matches: z.boolean().describe("True only if the image genuinely shows what the query describes — including actions and interactions, not just the objects in isolation."),
    score: z.number().describe("0.0 to 1.0. How well does the image match the query? Score 0.9+ only for perfect matches. Score 0.0-0.2 for images that contain the objects but do not depict the described action or interaction."),
    reason: z.string().describe("One short sentence explaining the score."),
});

async function scoreOne(query: string, imageBytes: Buffer) {
    return withQuotaRetry(
        () =>
            generateObject({
                model: vertex("gemini-2.5-flash"),
                schema: VlmScore,
                messages: [
                    {
                        role: "user",
                        content: [
                            {
                                type: "text",
                                text: `You are grading an image retrieval result. Be strict.

Query: "${query}"

Score 0.0 to 1.0. Rules:
- If the query describes an ACTION ("getting in", "holding", "carrying", "running", "entering"), the image must actually depict that action happening. Objects without the action = 0.0-0.15.
- If the query names a specific attribute ("red car", "man in blue"), the image must clearly show that attribute. Generic match = 0.1-0.2.
- A perfect depiction scores 0.9-1.0.
- A loosely related scene (right category, wrong specifics) scores 0.2-0.4.
- Completely unrelated scores 0.0.

Examples:
- Query "a guy getting in a car", image shows a man sitting on a couch = 0.05 (no car, no action).
- Query "a guy getting in a car", image shows a man standing next to a parked car = 0.2 (right objects, wrong action).
- Query "a guy getting in a car", image shows a man with one leg inside a car door, gripping the frame = 0.95 (action visible).

Return a strict score. False positives are the problem we are solving.`,
                            },
                            { type: "image", image: imageBytes, mediaType: "image/jpeg" },
                        ],
                    },
                ],
                abortSignal: AbortSignal.timeout(15000),
                temperature: 0,
            }),
        { label: "vlm-rerank" }
    );
}

export async function POST(req: NextRequest) {
    try {
        const body = await req.json();
        const parsed = BodySchema.safeParse(body);
        if (!parsed.success) {
            return NextResponse.json({ error: "invalid request" }, { status: 400 });
        }
        const { query, candidates } = parsed.data;

        if (!process.env.GOOGLE_PROJECT_ID || !process.env.GOOGLE_CLIENT_EMAIL || !process.env.GOOGLE_PRIVATE_KEY) {
            return NextResponse.json({ scored: candidates.map(c => ({ ...c, vlm_score: null, vlm_reason: "vlm-disabled" })) });
        }

        const results = await Promise.all(
            candidates.map(async (c) => {
                try {
                    const bytes = await fetchAndResize(c.photo_image_url);
                    if (!bytes) {
                        return { ...c, vlm_score: null, vlm_reason: "fetch-failed" };
                    }
                    const res = await scoreOne(query, bytes);
                    return {
                        ...c,
                        vlm_score: res.object.score,
                        vlm_matches: res.object.matches,
                        vlm_reason: res.object.reason,
                    };
                } catch (err: unknown) {
                    const e = err as { message?: string };
                    console.warn(`vlm-rerank failed for ${c.photo_id}:`, e?.message ?? err);
                    return { ...c, vlm_score: null, vlm_reason: "error" };
                }
            })
        );

        return NextResponse.json({ scored: results });
    } catch (err: unknown) {
        const e = err as { message?: string };
        console.error("vlm-rerank route error:", e?.message ?? err);
        return NextResponse.json({ error: "vlm-rerank failed" }, { status: 500 });
    }
}

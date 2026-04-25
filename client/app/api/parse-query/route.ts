import { NextRequest, NextResponse } from "next/server";
import { generateObject } from "ai";
import { z } from "zod";
import { vertex, withQuotaRetry } from "@/lib/vertex";

export const runtime = "nodejs";

const PlanSchema = z.object({
    mode: z.enum(["AND", "OR", "SINGLE"]).describe("AND = all clauses must match; OR = any clause matches; SINGLE = no structure, run the raw query"),
    clauses: z
        .array(
            z.object({
                object: z.string().describe("The bare head noun of the clause, e.g. 'car', 'dog', 'van'. DO NOT include attributes here."),
                attributes: z.array(z.string()).describe("All modifying attributes as separate strings: colors, materials, sizes, states, clothing. e.g. for 'red car' -> ['red']; for 'man in blue jacket carrying a bag' -> ['blue jacket', 'carrying a bag']. Empty array if none."),
                negated: z.boolean().describe("True if the clause is negated, e.g. 'not wearing a hat'"),
            })
        )
        .describe("One entry per atomic search clause. For SINGLE mode, return exactly one clause."),
});

export type SearchPlan = z.infer<typeof PlanSchema>;

const SYSTEM = `You convert a natural-language image-search query into a structured plan.

The plan is used by a retrieval pipeline. Downstream a vision-language model verifies actions and relationships, so DO NOT decompose action or relational queries into multiple entities. SigLIP ranks best on whole-phrase scenes — keep action phrases intact as one clause.

Mode rules:
- "X or Y", "X OR Y" -> mode:"OR", two clauses (true disjunction only)
- "not X", "without X", "no X" -> that clause has negated:true; keep mode of surrounding phrase
- Explicit conjunctions of DISTINCT scenes/subjects like "picture of a cat AND a dog where they are far apart" -> mode:"AND" — only when the user literally joins two independent entities
- Everything else — action queries, descriptive queries, attribute queries, single-noun queries -> mode:"SINGLE" with ONE clause containing the full phrase

Examples (SINGLE):
- "a guy getting in a car"           -> SINGLE: [{object:"a guy getting in a car", attributes:[]}]
- "woman holding a phone"            -> SINGLE: [{object:"woman holding a phone", attributes:[]}]
- "child in a red shirt and black cap" -> SINGLE: [{object:"child", attributes:["red shirt", "black cap"]}]
- "red car"                          -> SINGLE: [{object:"car", attributes:["red"]}]
- "ambulance"                        -> SINGLE: [{object:"ambulance", attributes:[]}]
- "white van on a highway"           -> SINGLE: [{object:"van on a highway", attributes:["white"]}]

Examples (OR / NOT):
- "dog or cat"                       -> OR: [{object:"dog"}, {object:"cat"}]
- "person not wearing a hat"         -> SINGLE with negated clause handled by downstream; keep positive intact

Object vs attribute extraction (ONLY when the object is a simple head noun):
- If the phrase is a simple noun with color/state modifiers, extract attributes: "red car" -> object:"car", attributes:["red"].
- If the phrase describes an action or scene, put the ENTIRE phrase in "object" with attributes:[].
- When in doubt, prefer putting more into "object" and less into attributes. Losing the action is worse than losing a descriptor.

Strip filler ("a photo of", "an image of", "picture of").`;

export async function POST(req: NextRequest) {
    try {
        const { query } = await req.json();
        if (!query || typeof query !== "string" || !query.trim()) {
            return NextResponse.json({ error: "query is required" }, { status: 400 });
        }

        if (!process.env.GOOGLE_PROJECT_ID || !process.env.GOOGLE_CLIENT_EMAIL || !process.env.GOOGLE_PRIVATE_KEY) {
            const plan: SearchPlan = { mode: "SINGLE", clauses: [{ object: query.trim(), attributes: [], negated: false }] };
            return NextResponse.json({ plan, fallback: "no-credentials" });
        }

        const result = await withQuotaRetry(
            () =>
                generateObject({
                    model: vertex("gemini-2.5-flash"),
                    schema: PlanSchema,
                    system: SYSTEM,
                    prompt: `Query: "${query}"`,
                    abortSignal: AbortSignal.timeout(8000),
                }),
            { label: "parse-query" }
        );

        return NextResponse.json({ plan: result.object });
    } catch (err: unknown) {
        const e = err as { message?: string };
        console.error("parse-query error:", e?.message ?? err);
        const body = await req.clone().json().catch(() => ({ query: "" }));
        const fallback: SearchPlan = { mode: "SINGLE", clauses: [{ object: String(body.query || ""), attributes: [], negated: false }] };
        return NextResponse.json({ plan: fallback, fallback: "error" });
    }
}

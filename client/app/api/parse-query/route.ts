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

Mode rules:
- "X or Y", "X OR Y" -> mode:"OR", two clauses
- "X and Y", "X with Y", "X near Y", "X next to Y", "X on Y" -> mode:"AND", one clause per object
- "not X", "without X", "no X" -> that clause has negated:true; keep it as a clause of the same mode
- Strip filler: "a photo of", "an image of", "picture showing", etc.
- Simple single noun phrase -> mode:"SINGLE" with one clause

Object vs attribute extraction (IMPORTANT):
- "object" is the BARE head noun only — the thing itself.
- "attributes" is an array containing every modifier: colors, materials, sizes, actions, clothing, adjacent items.
- Examples:
  * "red car"              -> object:"car", attributes:["red"]
  * "white van"            -> object:"van", attributes:["white"]
  * "man in blue jacket"   -> object:"man", attributes:["blue jacket"]
  * "black SUV with tinted windows" -> object:"SUV", attributes:["black", "tinted windows"]
  * "woman carrying a red backpack" -> object:"woman", attributes:["carrying a red backpack"]
  * "ambulance"            -> object:"ambulance", attributes:[]
- If the word is itself already a specific compound noun (e.g. "police car"), keep it intact as the object.`;

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

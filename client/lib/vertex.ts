import { createVertex } from "@ai-sdk/google-vertex";

function formatPrivateKey(key: string | undefined): string | undefined {
    if (!key) return undefined;
    return key
        .replace(/^"(.*)"$/, "$1")
        .replace(/\\n/g, "\n")
        .trim();
}

const PROJECT_ID = process.env.GOOGLE_PROJECT_ID;

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

export async function withQuotaRetry<T>(
    fn: () => Promise<T>,
    { label, maxAttempts = 4 }: { label: string; maxAttempts?: number } = { label: "vertex" }
): Promise<T> {
    let delay = 800;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await fn();
        } catch (err: unknown) {
            const e = err as { statusCode?: number; message?: string };
            const is429 = e?.statusCode === 429 || /429|quota|rate/i.test(e?.message ?? "");
            if (!is429 || attempt === maxAttempts) throw err;
            console.warn(`[${label}] 429 attempt ${attempt}, retrying in ${delay}ms`);
            await new Promise((r) => setTimeout(r, delay));
            delay *= 2;
        }
    }
    throw new Error(`[${label}] retry exhausted`);
}

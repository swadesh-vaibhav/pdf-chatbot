const BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000";

export type ChatContextChunk = {
  page: number;
  text: string;
  source: string;
};

export type ChatResponse = {
  answer: string;
  cached?: boolean;
  context?: ChatContextChunk[];
};

export type AutocompleteResponse = {
  suggestions: string[];
  cached?: boolean;
};

export async function uploadPdf(file: File) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE_URL}/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const detail = await safeError(res);
    throw new Error(detail || "Upload failed");
  }

  return res.json();
}

export async function askQuestion(query: string): Promise<ChatResponse> {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query }),
  });

  if (!res.ok) {
    const detail = await safeError(res);
    throw new Error(detail || "Chat request failed");
  }

  return res.json();
}

export async function getAutocomplete(prefix: string): Promise<AutocompleteResponse> {
  const res = await fetch(`${BASE_URL}/autocomplete`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ prefix }),
  });

  if (!res.ok) {
    const detail = await safeError(res);
    throw new Error(detail || "Autocomplete request failed");
  }

  return res.json();
}

async function safeError(res: Response): Promise<string | null> {
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") return data.detail;
    return JSON.stringify(data);
  } catch {
    try {
      return await res.text();
    } catch {
      return null;
    }
  }
}
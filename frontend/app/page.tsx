"use client";

import { useState } from "react";
import UploadPanel from "@/components/UploadPanel";
import ChatWindow, { type ChatMessage } from "@/components/ChatWindow";
import Composer from "@/components/Composer";
import { askQuestion, getAutocomplete, streamQuestion, uploadPdf } from "@/lib/api";

export default function Page() {
  const [fileName, setFileName] = useState("");
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  async function handleUpload(file: File) {
    try {
      setUploading(true);
      await uploadPdf(file);
      setFileName(file.name);
      setMessages([]);
    } catch (err) {
      console.error(err);
      alert(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handleSend(query: string) {
    try {
      setSending(true);
      setMessages((prev) => [...prev, { role: "user", content: query }]);

      const data = await askQuestion(query);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          context: data.context ?? [],
        },
      ]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: err instanceof Error ? err.message : "Something went wrong.",
        },
      ]);
    } finally {
      setSending(false);
    }
  }

  async function handleStreamSend(query: string) {
    setMessages((prev) => [
      ...prev,
      { role: "user", content: query },
      { role: "assistant", content: "", context: [] },
    ]);

    let assistantText = "";
    let assistantContext: any[] = [];

    try {
      await streamQuestion(
        query,
        (token: string) => {
          assistantText += token;
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant") {
              last.content = assistantText;
              last.context = assistantContext;
            }
            return [...next];
          });
        },
        (context: any[]) => {
          assistantContext = context;
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last?.role === "assistant") {
              last.context = assistantContext;
            }
            return [...next];
          });
        }
      );
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") {
          last.content = "Something went wrong while streaming the answer.";
        }
        return [...next];
      });
    }
  }

  async function handleAutocomplete(prefix: string) {
    const data = await getAutocomplete(prefix);
    return data.suggestions ?? [];
  }

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100">
      <div className="mx-auto flex min-h-screen max-w-5xl flex-col p-4">
        <UploadPanel
          fileName={fileName}
          uploading={uploading}
          onUpload={handleUpload}
        />

        <section className="mt-4 flex flex-1 flex-col rounded-2xl border border-neutral-800 bg-neutral-900">
          <ChatWindow messages={messages} sending={sending} />
          <Composer
            disabled={uploading || !fileName || sending}
            onSend={handleStreamSend}
            onAutocomplete={handleAutocomplete}
          />
        </section>
      </div>
    </main>
  );
}
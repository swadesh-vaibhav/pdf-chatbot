"use client";

import type { ChatContextChunk } from "@/lib/api";

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  context?: ChatContextChunk[];
};

type Props = {
  messages: ChatMessage[];
  sending: boolean;
};

export default function ChatWindow({ messages, sending }: Props) {
  return (
    <div className="flex-1 overflow-y-auto p-4">
      <div className="space-y-4">
        {messages.length === 0 ? (
          <div className="rounded-xl border border-dashed border-neutral-700 p-6 text-neutral-400">
            Upload a PDF, then ask questions. The input box will suggest grounded
            completions as you type.
          </div>
        ) : (
          messages.map((m, i) => (
            <MessageBubble key={i} message={m} />
          ))
        )}

        {sending ? (
          <div className="mr-auto max-w-3xl rounded-2xl bg-neutral-100 px-4 py-3 text-neutral-950">
            Thinking...
          </div>
        ) : null}
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div
      className={
        isUser
          ? "ml-auto max-w-3xl rounded-2xl bg-neutral-800 px-4 py-3 text-neutral-100"
          : "mr-auto max-w-3xl rounded-2xl bg-neutral-100 px-4 py-3 text-neutral-950"
      }
    >
      <div className="whitespace-pre-wrap leading-6">{message.content}</div>

      {!isUser && message.context && message.context.length > 0 ? (
        <div className="mt-3 border-t border-neutral-300 pt-3 text-sm">
          <div className="mb-2 font-semibold">Sources</div>
          <div className="space-y-2">
            {message.context.map((c, idx) => (
              <div key={idx} className="rounded-lg bg-neutral-200/70 px-3 py-2">
                <div className="mb-1 font-medium">Page {c.page}</div>
                <div className="text-neutral-700">
                  {c.text.length > 220 ? `${c.text.slice(0, 220)}...` : c.text}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
"use client";

import { useEffect, useRef, useState } from "react";
import SuggestionList from "@/components/SuggestionList";

type Props = {
  disabled?: boolean;
  onSend: (query: string) => void | Promise<void>;
  onAutocomplete: (prefix: string) => Promise<string[]>;
};

export default function Composer({ disabled, onSend, onAutocomplete }: Props) {
  const [input, setInput] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const [focused, setFocused] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const blurTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const prefix = input.trim();

    if (!prefix || disabled) {
      setSuggestions([]);
      return;
    }

    const timer = setTimeout(async () => {
      try {
        setLoadingSuggestions(true);
        const result = await onAutocomplete(prefix);
        setSuggestions(result.slice(0, 3));
      } catch {
        setSuggestions([]);
      } finally {
        setLoadingSuggestions(false);
      }
    }, 220);

    return () => clearTimeout(timer);
  }, [input, disabled, onAutocomplete]);

  async function sendNow() {
    const query = input.trim();
    if (!query || disabled) return;
    setSuggestions([]);
    setInput("");
    await onSend(query);
    textareaRef.current?.focus();
  }

  return (
    <div className="border-t border-neutral-800 p-4">
      <div className="relative">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onFocus={() => {
            if (blurTimeout.current) {
              clearTimeout(blurTimeout.current);
              blurTimeout.current = null;
            }
            setFocused(true);
          }}
          onBlur={() => {
            blurTimeout.current = setTimeout(() => {
              setFocused(false);
              blurTimeout.current = null;
            }, 150);
          }}
          placeholder="Ask a question about the PDF..."
          className="min-h-24 w-full resize-none rounded-2xl border border-neutral-700 bg-neutral-950 px-4 py-3 text-neutral-100 outline-none ring-0 placeholder:text-neutral-500"
          disabled={disabled}
          onKeyDown={async (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              await sendNow();
            }
          }}
        />

        {focused ? (
          <SuggestionList
            suggestions={suggestions}
            loading={loadingSuggestions}
            onSelect={(s) => {
              setInput(s);
              textareaRef.current?.focus();
            }}
          />
        ) : null}
      </div>

      <div className="mt-3 flex items-center justify-between gap-4">
        <div className="text-sm text-neutral-400">
          {disabled ? "Upload a PDF first." : "Enter to send, Shift+Enter for a new line."}
        </div>

        <button
          type="button"
          onClick={sendNow}
          disabled={disabled || !input.trim()}
          className="rounded-xl bg-white px-4 py-2 text-sm font-medium text-black transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </div>
  );
}
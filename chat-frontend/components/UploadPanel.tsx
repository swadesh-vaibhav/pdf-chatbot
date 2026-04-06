"use client";

import { useRef } from "react";

type Props = {
  fileName: string;
  uploading: boolean;
  onUpload: (file: File) => void | Promise<void>;
};

export default function UploadPanel({ fileName, uploading, onUpload }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  return (
    <div className="flex items-center justify-between gap-4 rounded-2xl border border-neutral-800 bg-neutral-900 px-4 py-3">
      <div className="min-w-0">
        <div className="text-lg font-semibold text-neutral-100">PDF Chatbot</div>
        <div className="truncate text-sm text-neutral-400">
          {fileName ? `Loaded: ${fileName}` : "No PDF loaded"}
        </div>
      </div>

      <div className="flex items-center gap-3">
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            await onUpload(file);
            e.target.value = "";
          }}
        />

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="rounded-xl bg-white px-4 py-2 text-sm font-medium text-black transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          disabled={uploading}
        >
          {uploading ? "Uploading..." : "Upload PDF"}
        </button>
      </div>
    </div>
  );
}
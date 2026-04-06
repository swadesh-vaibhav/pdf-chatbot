"use client";

type Props = {
  suggestions: string[];
  loading: boolean;
  onSelect: (suggestion: string) => void;
};

export default function SuggestionList({ suggestions, loading, onSelect }: Props) {
  if (loading && suggestions.length === 0) {
    return (
      <div className="mt-2 rounded-2xl border border-neutral-800 bg-neutral-950 p-3 text-sm text-neutral-400">
        Finding grounded completions...
      </div>
    );
  }

  if (suggestions.length === 0) return null;

  return (
    <div className="mt-2 rounded-2xl border border-neutral-800 bg-neutral-950 p-2 shadow-lg">
      <div className="px-2 pb-2 text-xs uppercase tracking-wide text-neutral-500">
        Autocomplete
      </div>

      <div className="space-y-1">
        {suggestions.map((s, i) => (
          <button
            key={`${s}-${i}`}
            type="button"
            onClick={() => onSelect(s)}
            className="block w-full rounded-xl px-3 py-2 text-left text-sm text-neutral-100 transition hover:bg-neutral-800"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
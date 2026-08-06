import { ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';
import {
  HORIZONTAL_POSE_LABELS,
  VERTICAL_POSE_LABELS,
  EXPRESSION_LABELS,
  LIGHTING_LABELS,
  QUALITY_LABELS,
} from '../types/taxonomy';
import type { HorizontalPose, VerticalPose, ExpressionLabel, LightingLabel, QualityLevel } from '../types/taxonomy';

interface FilterSectionProps<T extends string> {
  title: string;
  options: { value: T; label: string; count?: number }[];
  selected: T[];
  onChange: (values: T[]) => void;
}

function FilterSection<T extends string>({
  title,
  options,
  selected,
  onChange,
}: FilterSectionProps<T>) {
  const [expanded, setExpanded] = useState(true);

  const toggle = (value: T) => {
    if (selected.includes(value)) {
      onChange(selected.filter((v) => v !== value));
    } else {
      onChange([...selected, value]);
    }
  };

  return (
    <div className="border-b border-border">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between py-3 px-1 text-sm text-text-secondary hover:text-text transition-colors focus:outline-none focus:ring-2 focus:ring-border-hover focus:ring-inset rounded"
        aria-expanded={expanded}
      >
        {title}
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>
      {expanded && (
        <div className="pb-3 space-y-0.5">
          {options.map((opt) => (
            <button
              key={opt.value}
              onClick={() => toggle(opt.value)}
              className={`w-full flex items-center justify-between px-1 py-1.5 text-sm rounded-md transition-colors focus:outline-none focus:ring-2 focus:ring-border-hover focus:ring-inset ${
                selected.includes(opt.value)
                  ? 'text-text'
                  : 'text-text-secondary hover:text-text'
              }`}
              role="checkbox"
              aria-checked={selected.includes(opt.value)}
            >
              <span className="flex items-center gap-2">
                <span
                  className={`w-3.5 h-3.5 rounded border flex items-center justify-center transition-colors ${
                    selected.includes(opt.value)
                      ? 'bg-text border-text'
                      : 'border-border-hover'
                  }`}
                >
                  {selected.includes(opt.value) && (
                    <svg
                      width="8"
                      height="6"
                      viewBox="0 0 8 6"
                      fill="none"
                      className="text-bg"
                    >
                      <path
                        d="M1 3L3 5L7 1"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  )}
                </span>
                {opt.label}
              </span>
              {opt.count !== undefined && (
                <span className="text-xs text-text-muted tabular-nums">
                  {opt.count}
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

interface FiltersPanelProps {
  filters: Record<string, string>;
  onChange: (filters: Record<string, string>) => void;
}

const HORIZONTAL_POSE_OPTIONS = Object.entries(HORIZONTAL_POSE_LABELS).map(
  ([value, label]) => ({ value: value as HorizontalPose, label })
);

const VERTICAL_POSE_OPTIONS = Object.entries(VERTICAL_POSE_LABELS).map(
  ([value, label]) => ({ value: value as VerticalPose, label })
);

const EXPRESSION_OPTIONS = Object.entries(EXPRESSION_LABELS).map(
  ([value, label]) => ({ value: value as ExpressionLabel, label })
);

const LIGHTING_OPTIONS = Object.entries(LIGHTING_LABELS).map(
  ([value, label]) => ({ value: value as LightingLabel, label })
);

const QUALITY_OPTIONS = Object.entries(QUALITY_LABELS).map(
  ([value, label]) => ({ value: value as QualityLevel, label })
);

export function FiltersPanel({ filters, onChange }: FiltersPanelProps) {
  const setFilter = (key: string, values: string[]) => {
    const next = { ...filters };
    if (values.length === 0) {
      delete next[key];
    } else {
      next[key] = values.join(',');
    }
    onChange(next);
  };

  const parseMulti = (key: string): string[] => {
    const val = filters[key];
    return val ? val.split(',') : [];
  };

  return (
    <div className="space-y-1">
      <h3 className="text-xs font-medium text-text-muted uppercase tracking-widest mb-3">
        Filters
      </h3>

      <FilterSection
        title="Horizontal Angle"
        options={HORIZONTAL_POSE_OPTIONS}
        selected={parseMulti('horizontal_pose') as HorizontalPose[]}
        onChange={(v) => setFilter('horizontal_pose', v)}
      />

      <FilterSection
        title="Vertical Pose"
        options={VERTICAL_POSE_OPTIONS}
        selected={parseMulti('vertical_pose') as VerticalPose[]}
        onChange={(v) => setFilter('vertical_pose', v)}
      />

      <FilterSection
        title="Expression"
        options={EXPRESSION_OPTIONS}
        selected={parseMulti('expression') as ExpressionLabel[]}
        onChange={(v) => setFilter('expression', v)}
      />

      <FilterSection
        title="Lighting"
        options={LIGHTING_OPTIONS}
        selected={parseMulti('lighting') as LightingLabel[]}
        onChange={(v) => setFilter('lighting', v)}
      />

      <FilterSection
        title="Quality"
        options={QUALITY_OPTIONS}
        selected={parseMulti('quality') as QualityLevel[]}
        onChange={(v) => setFilter('quality', v)}
      />

      {Object.keys(filters).length > 0 && (
        <button
          onClick={() => onChange({})}
          className="w-full py-2 text-sm text-text-muted hover:text-text transition-colors focus:outline-none focus:ring-2 focus:ring-border-hover rounded"
        >
          Clear all filters
        </button>
      )}
    </div>
  );
}

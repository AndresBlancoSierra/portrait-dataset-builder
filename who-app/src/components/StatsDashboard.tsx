import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { StatsSkeleton, EmptyState } from './Skeletons';
import { BarChart3 } from 'lucide-react';

function StatCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: string | number;
  sub?: string;
}) {
  return (
    <div className="p-3 bg-bg-card rounded-lg">
      <p className="text-xs text-text-muted uppercase tracking-wider mb-1">
        {label}
      </p>
      <p className="text-2xl font-light tabular-nums">{value}</p>
      {sub && <p className="text-xs text-text-muted mt-0.5">{sub}</p>}
    </div>
  );
}

function BarChart({
  data,
  maxItems = 8,
}: {
  data: Record<string, number>;
  maxItems?: number;
}) {
  const entries = Object.entries(data)
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxItems);
  const maxVal = Math.max(...entries.map((e) => e[1]), 1);

  return (
    <div className="space-y-1.5">
      {entries.map(([label, count]) => (
        <div key={label} className="flex items-center gap-2">
          <span className="text-xs text-text-secondary w-20 truncate text-right">
            {label}
          </span>
          <div className="flex-1 h-2 bg-bg-card rounded-full overflow-hidden">
            <div
              className="h-full bg-text-secondary rounded-full transition-all"
              style={{ width: `${(count / maxVal) * 100}%` }}
            />
          </div>
          <span className="text-xs text-text-muted tabular-nums w-8 text-right">
            {count}
          </span>
        </div>
      ))}
    </div>
  );
}

export function StatsDashboard({ libraryName }: { libraryName: string }) {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['stats', libraryName],
    queryFn: () => api.getStats(libraryName),
    enabled: !!libraryName,
  });

  if (isLoading) return <StatsSkeleton />;

  if (!stats) {
    return (
      <EmptyState
        icon={<BarChart3 size={32} />}
        title="No stats available"
        description="Build a library to see statistics."
      />
    );
  }

  return (
    <div className="space-y-6">
      <h3 className="text-xs font-medium text-text-muted uppercase tracking-widest">
        Statistics
      </h3>

      <div className="grid grid-cols-2 gap-3">
        <StatCard
          label="Total"
          value={stats.total_images.toLocaleString()}
        />
        <StatCard
          label="Verified"
          value={stats.verified_images.toLocaleString()}
          sub={`${Math.round((stats.verified_images / Math.max(stats.total_images, 1)) * 100)}%`}
        />
        <StatCard
          label="Avg Quality"
          value={Math.round(stats.avg_quality * 100)}
        />
        <StatCard
          label="Avg Yaw"
          value={`${Math.round(stats.avg_yaw)}°`}
        />
      </div>

      {Object.keys(stats.expressions).length > 0 && (
        <div>
          <p className="text-xs text-text-muted uppercase tracking-wider mb-2">
            Expressions
          </p>
          <BarChart data={stats.expressions} />
        </div>
      )}

      {Object.keys(stats.angles).length > 0 && (
        <div>
          <p className="text-xs text-text-muted uppercase tracking-wider mb-2">
            Angles
          </p>
          <BarChart data={stats.angles} />
        </div>
      )}
    </div>
  );
}

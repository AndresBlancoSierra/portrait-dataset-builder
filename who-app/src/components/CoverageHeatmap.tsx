import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { HeatmapSkeleton, EmptyState } from './Skeletons';
import { Layers } from 'lucide-react';

function cellColor(value: number, max: number): string {
  if (max === 0) return 'bg-bg-card';
  const intensity = Math.min(value / max, 1);
  if (intensity === 0) return 'bg-bg-card';
  return 'bg-white';
}

export function CoverageHeatmap({ libraryName }: { libraryName: string }) {
  const { data: coverage, isLoading } = useQuery({
    queryKey: ['coverage', libraryName],
    queryFn: () => api.getCoverage(libraryName),
    enabled: !!libraryName,
  });

  if (isLoading) return <HeatmapSkeleton />;

  if (!coverage) {
    return (
      <EmptyState
        icon={<Layers size={32} />}
        title="No coverage data"
        description="Build a library to see pose coverage."
      />
    );
  }

  const { yaw_bins, pitch_bins, heatmap } = coverage;
  const maxVal = Math.max(...heatmap.flat(), 1);

  const yawLabels = yaw_bins.map((v) =>
    v === 0 ? '0°' : `${v > 0 ? '+' : ''}${v}°`
  );
  const pitchLabels = pitch_bins.map((v) =>
    v === 0 ? '0°' : `${v > 0 ? '+' : ''}${v}°`
  );

  return (
    <div>
      <h3 className="text-xs font-medium text-text-muted uppercase tracking-widest mb-4">
        Pose Coverage
      </h3>

      <div className="flex items-end gap-0">
        <div className="flex flex-col items-end pr-2 gap-0">
          {yawLabels.map((label) => (
            <span
              key={label}
              className="text-[10px] text-text-muted h-6 flex items-center"
            >
              {label}
            </span>
          ))}
        </div>

        <div className="flex flex-col gap-0">
          <div className="flex gap-0">
            {pitchLabels.map((label) => (
              <span
                key={label}
                className="text-[10px] text-text-muted w-8 h-4 flex items-center justify-center"
              >
                {label}
              </span>
            ))}
          </div>

          {heatmap.map((row, yIdx) => (
            <div key={yIdx} className="flex gap-0">
              {row.map((val, pIdx) => (
                <div
                  key={pIdx}
                  className={`w-8 h-6 rounded-[2px] transition-colors ${cellColor(
                    val,
                    maxVal
                  )}`}
                  style={{
                    opacity: val === 0 ? 0.3 : undefined,
                  }}
                  title={`Yaw ${yawLabels[yIdx]}, Pitch ${pitchLabels[pIdx]}: ${val} images`}
                  role="gridcell"
                  aria-label={`Yaw ${yawLabels[yIdx]}, Pitch ${pitchLabels[pIdx]}: ${val} images`}
                >
                  {val > 0 && (
                    <span className="text-[9px] text-text-secondary flex items-center justify-center h-full">
                      {val}
                    </span>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-2 mt-3 text-[10px] text-text-muted">
        <span>Less</span>
        <div className="flex gap-0.5">
          {[0.15, 0.3, 0.5, 0.7, 1].map((o) => (
            <div
              key={o}
              className="w-4 h-2 rounded-sm bg-white"
              style={{ opacity: o }}
            />
          ))}
        </div>
        <span>More</span>
      </div>
    </div>
  );
}

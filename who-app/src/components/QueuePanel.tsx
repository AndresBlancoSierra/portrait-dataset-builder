import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import type { QueueJob } from '../types';
import { Pause, Play, Trash2, RotateCcw, Loader2 } from 'lucide-react';

function QueueItem({ job }: { job: QueueJob }) {
  const queryClient = useQueryClient();

  const handleCancel = async () => {
    await api.cancelQueueJob(job.id);
    queryClient.invalidateQueries({ queryKey: ['buildQueue'] });
    queryClient.invalidateQueries({ queryKey: ['libraries'] });
  };

  const handleRetry = async () => {
    await api.retryQueueJob(job.id);
    queryClient.invalidateQueries({ queryKey: ['buildQueue'] });
  };

  const handleRemove = async () => {
    await api.removeQueueJob(job.id);
    queryClient.invalidateQueries({ queryKey: ['buildQueue'] });
    queryClient.invalidateQueries({ queryKey: ['libraries'] });
  };

  return (
    <div className="flex items-center gap-3 py-2.5 px-3 rounded-md hover:bg-gray-50 group">
      <div className="w-5 flex-shrink-0">
        {job.status === 'running' ? (
          <Loader2 size={14} className="text-gray-900 animate-spin" />
        ) : (
          <span className="text-xs text-gray-400 font-mono">
            {job.position != null ? `#${job.position}` : '—'}
          </span>
        )}
      </div>

      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-900 truncate">{job.name}</p>
        {job.status === 'running' && job.stage_label && (
          <div className="flex items-center gap-2 mt-1">
            <p className="text-xs text-gray-500">{job.stage_label}</p>
            {job.total != null && job.total > 0 && (
              <p className="text-xs text-gray-400">
                {job.processed ?? 0} / {job.total}
              </p>
            )}
          </div>
        )}
        {job.status === 'queued' && (
          <p className="text-xs text-gray-400 mt-0.5">Waiting</p>
        )}
        {job.status === 'failed' && (
          <p className="text-xs text-red-500 mt-0.5 truncate">{job.error || 'Failed'}</p>
        )}
      </div>

      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        {job.status === 'queued' && (
          <button onClick={handleRemove} className="p-1 text-gray-400 hover:text-red-500" title="Remove">
            <Trash2 size={13} />
          </button>
        )}
        {job.status === 'failed' && (
          <button onClick={handleRetry} className="p-1 text-gray-400 hover:text-gray-700" title="Retry">
            <RotateCcw size={13} />
          </button>
        )}
        {job.status === 'running' && (
          <button onClick={handleCancel} className="p-1 text-gray-400 hover:text-red-500" title="Cancel">
            <Trash2 size={13} />
          </button>
        )}
      </div>
    </div>
  );
}

export default function QueuePanel() {
  const queryClient = useQueryClient();
  const { data } = useQuery({
    queryKey: ['buildQueue'],
    queryFn: () => api.getBuildQueue(),
    refetchInterval: (query) => {
      const d = query.state.data;
      if (!d) return 2000;
      const hasActive = d.jobs.some((j) => j.status === 'running' || j.status === 'queued');
      return hasActive ? 2000 : false;
    },
  });

  if (!data || data.jobs.length === 0) return null;

  const handlePause = async () => {
    if (data.queue_paused) {
      await api.resumeQueue();
    } else {
      await api.pauseQueue();
    }
    queryClient.invalidateQueries({ queryKey: ['buildQueue'] });
  };

  return (
    <div className="mt-8">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-900">Build Queue</h2>
        <button
          onClick={handlePause}
          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700"
        >
          {data.queue_paused ? <Play size={12} /> : <Pause size={12} />}
          {data.queue_paused ? 'Resume' : 'Pause'}
        </button>
      </div>
      <div className="border border-gray-100 rounded-lg divide-y divide-gray-50">
        {data.jobs.map((job) => (
          <QueueItem key={job.id} job={job} />
        ))}
      </div>
    </div>
  );
}

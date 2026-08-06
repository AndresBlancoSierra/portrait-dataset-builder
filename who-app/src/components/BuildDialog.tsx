import { useState, useEffect, useCallback } from 'react';
import { useMutation } from '@tanstack/react-query';
import { X, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { api } from '../api/client';
import type { PipelineProgress } from '../types';

const STAGE_LABELS: Record<string, string> = {
  search: 'Searching',
  download: 'Downloading',
  face_detection: 'Detecting faces',
  face_verification: 'Verifying identity',
  semantic_filter: 'Filtering non-portraits',
  quality: 'Scoring quality',
  duplicates: 'Removing duplicates',
  classification: 'Classifying',
  export: 'Exporting',
};

export function BuildDialog({
  identity,
  onClose,
  onComplete,
}: {
  identity: string;
  onClose: () => void;
  onComplete: (name: string) => void;
}) {
  const [progress, setProgress] = useState<PipelineProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  const buildMut = useMutation({
    mutationFn: () => api.startBuild(identity),
    onError: (err: Error) => setError(err.message),
  });

  const pollProgress = useCallback(async (taskId: string) => {
    try {
      const p = await api.getBuildProgress(taskId);
      setProgress(p);
      if (p.status === 'completed') {
        onComplete(identity);
      } else if (p.status === 'failed') {
        setError('Build failed');
      } else {
        setTimeout(() => pollProgress(taskId), 1000);
      }
    } catch (err: any) {
      setError(err.message);
    }
  }, [identity, onComplete]);

  useEffect(() => {
    if (buildMut.data) {
      pollProgress(buildMut.data.task_id);
    }
  }, [buildMut.data, pollProgress]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const pct = progress
    ? progress.items_total > 0
      ? Math.round((progress.items_processed / progress.items_total) * 100)
      : 0
    : 0;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
        onClick={onClose}
        role="dialog"
        aria-modal="true"
        aria-label="Build library"
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
          className="bg-bg-card rounded-xl shadow-xl w-full max-w-md mx-4 overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between px-6 py-4 border-b border-border">
            <h2 className="text-sm font-semibold text-text-secondary">Build Library</h2>
            <button
              onClick={onClose}
              className="p-1.5 rounded-md text-text-muted hover:text-text hover:bg-bg-hover transition-colors"
              aria-label="Close dialog"
            >
              <X size={16} />
            </button>
          </div>

          <div className="p-6">
            {/* Identity */}
            <div className="mb-5">
              <p className="text-sm text-text-secondary">
                Identity:{' '}
                <span className="text-text font-semibold">{identity}</span>
              </p>
            </div>

            {/* Error */}
            {error && (
              <div className="mb-4 p-3 bg-danger/10 border border-danger/20 rounded-lg text-sm text-danger">
                {error}
              </div>
            )}

            {/* Progress */}
            {progress && (
              <div className="mb-5">
                <div className="flex items-center justify-between mb-2.5">
                  <span className="text-sm text-text-secondary">
                    {progress.stage_label || STAGE_LABELS[progress.stage] || progress.stage}
                  </span>
                  <span className="text-xs text-text-muted tabular-nums">
                    {progress.items_processed.toLocaleString()} / {progress.items_total.toLocaleString()}
                  </span>
                </div>
                <div className="h-1.5 bg-white rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-accent rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${pct}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
              </div>
            )}
          </div>

          <div className="flex justify-end gap-2 px-6 py-4 border-t border-border bg-bg-card">
            <button
              onClick={onClose}
              className="px-5 py-2 text-sm text-text-secondary hover:text-text rounded-lg border border-border hover:bg-bg-elevated transition-colors"
            >
              Cancel
            </button>
            {!buildMut.data && (
              <button
                onClick={() => buildMut.mutate()}
                disabled={buildMut.isPending}
                className="px-5 py-2 text-sm font-medium bg-accent text-white rounded-lg hover:opacity-90 disabled:opacity-40 transition-opacity flex items-center gap-2"
              >
                {buildMut.isPending ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  'Start Build'
                )}
              </button>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

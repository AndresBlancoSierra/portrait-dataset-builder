import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import { Check, X, ArrowRight, CheckCircle } from 'lucide-react';
import { api } from '../api/client';
import { ReviewSkeleton, EmptyState } from './Skeletons';

export function ReviewPanel({ libraryName }: { libraryName: string }) {
  const queryClient = useQueryClient();
  const [currentIdx, setCurrentIdx] = useState(0);

  const { data: queue = [], isLoading } = useQuery({
    queryKey: ['review', libraryName],
    queryFn: () => api.getReviewQueue(libraryName),
    enabled: !!libraryName,
  });

  const reviewMut = useMutation({
    mutationFn: ({ hash, accepted }: { hash: string; accepted: boolean }) =>
      api.reviewImage(libraryName, hash, accepted),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['review', libraryName] });
      setCurrentIdx((prev) => Math.max(0, Math.min(prev, queue.length - 2)));
    },
  });

  if (isLoading) return <ReviewSkeleton />;

  if (queue.length === 0) {
    return (
      <EmptyState
        icon={<CheckCircle size={32} />}
        title="All caught up"
        description="No images in the review queue."
      />
    );
  }

  const current = queue[Math.max(0, Math.min(currentIdx, queue.length - 1))];
  if (!current) return null;

  return (
    <div>
      <h3 className="text-xs font-medium text-text-muted uppercase tracking-widest mb-4">
        Review Queue
      </h3>

      <div className="text-sm text-text-secondary mb-3">
        {currentIdx + 1} / {queue.length}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={current.content_hash}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.15 }}
          className="aspect-[3/4] rounded-lg overflow-hidden bg-bg-card border border-border mb-3"
        >
          <img
            src={api.getImageUrl(libraryName, current.content_hash)}
            alt=""
            className="w-full h-full object-cover"
          />
        </motion.div>
      </AnimatePresence>

      <div className="flex gap-2">
        <button
          onClick={() =>
            reviewMut.mutate({ hash: current.content_hash, accepted: false })
          }
          disabled={reviewMut.isPending}
          className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg border border-danger/30 text-danger hover:bg-danger/10 transition-colors text-sm focus:outline-none focus:ring-2 focus:ring-danger/50 focus:ring-offset-2 focus:ring-offset-bg"
          aria-label="Reject image"
        >
          <X size={16} />
          Reject
        </button>
        <button
          onClick={() => {
            setCurrentIdx((prev) => Math.min(prev + 1, queue.length - 1));
          }}
          className="px-3 py-2.5 rounded-lg border border-border hover:bg-bg-hover transition-colors text-text-muted focus:outline-none focus:ring-2 focus:ring-border-hover focus:ring-offset-2 focus:ring-offset-bg"
          aria-label="Skip image"
        >
          <ArrowRight size={16} />
        </button>
        <button
          onClick={() =>
            reviewMut.mutate({ hash: current.content_hash, accepted: true })
          }
          disabled={reviewMut.isPending}
          className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg border border-success/30 text-success hover:bg-success/10 transition-colors text-sm focus:outline-none focus:ring-2 focus:ring-success/50 focus:ring-offset-2 focus:ring-offset-bg"
          aria-label="Accept image"
        >
          <Check size={16} />
          Accept
        </button>
      </div>

      {current.quality && (
        <div className="mt-3 p-3 bg-bg-card rounded-lg text-xs space-y-1">
          <div className="flex justify-between">
            <span className="text-text-muted">Quality</span>
            <span className="text-text-secondary tabular-nums">
              {Math.round(current.quality.final_score * 100)}
            </span>
          </div>
          {current.face && (
            <div className="flex justify-between">
              <span className="text-text-muted">Yaw</span>
              <span className="text-text-secondary tabular-nums">
                {Math.round(current.face.yaw)}°
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

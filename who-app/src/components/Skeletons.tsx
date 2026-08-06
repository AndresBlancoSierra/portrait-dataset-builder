import { motion } from 'framer-motion';

export function SkeletonPulse({ className = '' }: { className?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0.5 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1, repeat: Infinity, repeatType: 'reverse' }}
      className={`bg-bg-card rounded-lg ${className}`}
    />
  );
}

export function GallerySkeleton() {
  return (
    <div className="h-full p-4">
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-0">
        {Array.from({ length: 20 }).map((_, i) => (
          <div key={i} className="p-1.5">
            <SkeletonPulse className="aspect-[3/4] rounded-lg" />
          </div>
        ))}
      </div>
    </div>
  );
}

export function LibraryCardSkeleton() {
  return (
    <div className="bg-bg-card border border-border rounded-xl overflow-hidden">
      <SkeletonPulse className="aspect-[16/10] rounded-none" />
      <div className="p-3.5 space-y-2.5">
        <div className="space-y-1.5">
          <SkeletonPulse className="h-3.5 w-28" />
          <SkeletonPulse className="h-2.5 w-16" />
        </div>
        <div className="flex gap-3">
          <SkeletonPulse className="h-[3px] flex-1" />
          <SkeletonPulse className="h-[3px] flex-1" />
        </div>
      </div>
    </div>
  );
}

export function ReviewSkeleton() {
  return (
    <div className="space-y-3">
      <SkeletonPulse className="h-3 w-24" />
      <SkeletonPulse className="aspect-[3/4] rounded-lg" />
      <div className="flex gap-2">
        <SkeletonPulse className="h-10 flex-1" />
        <SkeletonPulse className="h-10 flex-1" />
      </div>
    </div>
  );
}

export function StatsSkeleton() {
  return (
    <div className="space-y-4">
      <SkeletonPulse className="h-3 w-20" />
      <div className="grid grid-cols-2 gap-3">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="p-3 bg-bg-card rounded-lg space-y-2">
            <SkeletonPulse className="h-2.5 w-12" />
            <SkeletonPulse className="h-6 w-16" />
          </div>
        ))}
      </div>
      <div className="space-y-2">
        <SkeletonPulse className="h-2.5 w-16" />
        {Array.from({ length: 4 }).map((_, i) => (
          <SkeletonPulse key={i} className="h-2 w-full" />
        ))}
      </div>
    </div>
  );
}

export function HeatmapSkeleton() {
  return (
    <div className="space-y-4">
      <SkeletonPulse className="h-3 w-28" />
      <div className="grid grid-cols-6 gap-0.5">
        {Array.from({ length: 108 }).map((_, i) => (
          <SkeletonPulse key={i} className="w-8 h-6" />
        ))}
      </div>
    </div>
  );
}

export function EmptyState({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center py-16 text-center"
    >
      <div className="text-text-muted mb-3">{icon}</div>
      <h3 className="text-sm font-medium text-text-secondary mb-1">{title}</h3>
      {description && (
        <p className="text-xs text-text-muted max-w-xs">{description}</p>
      )}
    </motion.div>
  );
}

import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Loader2, CheckCircle2, AlertCircle, RotateCw, Clock } from 'lucide-react';
import { api } from '../api/client';
import { GANDALF_FACTS } from '../data/gandalfFacts';

const ALL_STAGES = [
  'search',
  'url_safety_filter',
  'identity_bootstrap',
  'download',
  'safety_gate',
  'face_detection',
  'face_verification',
  'semantic_filter',
  'quality',
  'duplicates',
  'classification',
  'export',
];

const STAGE_LABELS: Record<string, string> = {
  search: 'Searching',
  url_safety_filter: 'Filtering unsafe URLs',
  identity_bootstrap: 'Discovering identity',
  download: 'Downloading',
  safety_gate: 'Checking content safety',
  face_detection: 'Detecting faces',
  face_verification: 'Verifying identity',
  semantic_filter: 'Filtering non-portraits',
  quality: 'Scoring quality',
  duplicates: 'Removing duplicates',
  classification: 'Classifying',
  export: 'Exporting',
};

function shuffleArray<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function GandalfFact({ stage: _stage }: { stage: string }) {
  const [shuffledFacts, setShuffledFacts] = useState<string[]>(() => shuffleArray(GANDALF_FACTS));
  const [factIndex, setFactIndex] = useState(0);
  const [fadeIn, setFadeIn] = useState(true);

  useEffect(() => {
    const interval = setInterval(() => {
      setFadeIn(false);
      setTimeout(() => {
        setFactIndex((prev) => {
          if (prev >= shuffledFacts.length - 1) {
            setShuffledFacts(shuffleArray(GANDALF_FACTS));
            return 0;
          }
          return prev + 1;
        });
        setFadeIn(true);
      }, 400);
    }, 8000 + Math.random() * 4000);

    return () => clearInterval(interval);
  }, [shuffledFacts]);

  const fact = shuffledFacts[factIndex % shuffledFacts.length];

  return (
    <div style={{ marginTop: 20, textAlign: 'center' }}>
      <p style={{ fontSize: 10, color: '#a3a3a3', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8 }}>
        Did you know?
      </p>
      <AnimatePresence mode="wait">
        <motion.p
          key={fact}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: fadeIn ? 1 : 0, y: fadeIn ? 0 : -4 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.4 }}
          style={{ fontSize: 13, color: '#525252', lineHeight: 1.5, fontStyle: 'italic', maxWidth: 360, margin: '0 auto' }}
        >
          "{fact}"
        </motion.p>
      </AnimatePresence>
    </div>
  );
}

function StageIndicator({
  stageName,
  currentStage,
  stagesCompleted,
}: {
  stageName: string;
  currentStage: string | null;
  stagesCompleted: string[];
}) {
  const isCompleted = stagesCompleted.includes(stageName);
  const isCurrent = currentStage === stageName;

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '6px 0' }}>
      <div style={{ width: 20, display: 'flex', justifyContent: 'center' }}>
        {isCompleted ? (
          <CheckCircle2 size={14} style={{ color: '#16a34a' }} />
        ) : isCurrent ? (
          <Loader2 size={14} className="animate-spin" style={{ color: '#0a0a0a' }} />
        ) : (
          <div style={{ width: 14, height: 14, borderRadius: '50%', border: '1.5px solid #e5e5e5' }} />
        )}
      </div>
      <span
        style={{
          fontSize: 12,
          color: isCompleted ? '#16a34a' : isCurrent ? '#0a0a0a' : '#a3a3a3',
          fontWeight: isCurrent ? 600 : 400,
        }}
      >
        {STAGE_LABELS[stageName] || stageName}
      </span>
    </div>
  );
}

export default function BuildProgressPage() {
  const { name } = useParams<{ name: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const libraryName = decodeURIComponent(name || '');

  const { data: progress, isLoading, error } = useQuery({
    queryKey: ['buildProgress', libraryName],
    queryFn: () => api.getBuildProgress(libraryName),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 1000;
      if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
        return false;
      }
      if (data.library_status === 'queued') {
        return 2000;
      }
      return 1500;
    },
    retry: 2,
    retryDelay: 1000,
  });

  const cancelMut = useMutation({
    mutationFn: () => api.cancelBuild(libraryName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['buildProgress', libraryName] });
      queryClient.invalidateQueries({ queryKey: ['libraries'] });
    },
  });

  const restartMut = useMutation({
    mutationFn: () => api.startBuild(libraryName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['buildProgress', libraryName] });
      queryClient.invalidateQueries({ queryKey: ['libraries'] });
    },
  });

  useEffect(() => {
    if (progress?.status === 'completed' || progress?.status === 'failed' || progress?.status === 'cancelled') {
      queryClient.invalidateQueries({ queryKey: ['libraries'] });
    }
  }, [progress?.status, libraryName, queryClient]);

  const stagesCompleted = useMemo(() => {
    if (!progress) return [];
    if (progress.status === 'completed') return [...ALL_STAGES];
    const completed: string[] = [];
    const currentIdx = progress.current_stage ? ALL_STAGES.indexOf(progress.current_stage) : -1;
    for (let i = 0; i < currentIdx; i++) {
      completed.push(ALL_STAGES[i]);
    }
    return completed;
  }, [progress]);

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 size={24} className="animate-spin" style={{ color: '#a3a3a3' }} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4">
        <AlertCircle size={24} style={{ color: '#dc2626' }} />
        <p style={{ fontSize: 14, color: '#525252' }}>Failed to load build progress</p>
        <p style={{ fontSize: 12, color: '#a3a3a3' }}>{error.message}</p>
        <button
          onClick={() => navigate('/')}
          style={{
            padding: '10px 20px',
            fontSize: 13,
            borderRadius: 8,
            border: '1px solid #e5e5e5',
            backgroundColor: 'transparent',
            color: '#525252',
            cursor: 'pointer',
            marginTop: 16,
          }}
        >
          Back to Home
        </button>
      </div>
    );
  }

  if (!progress) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4">
        <Loader2 size={24} className="animate-spin" style={{ color: '#a3a3a3' }} />
        <p style={{ fontSize: 13, color: '#a3a3a3' }}>Build starting...</p>
      </div>
    );
  }

  const pct = progress.items_total > 0
    ? Math.round((progress.items_processed / progress.items_total) * 100)
    : 0;

  const isRunning = (progress.status === 'running' || progress.status === 'pending') && !isQueued;
  const isQueued = progress.library_status === 'queued';
  const isFailed = progress.status === 'failed';
  const isCancelled = progress.status === 'cancelled';
  const isCompleted = progress.status === 'completed';
  const isEmpty = progress.library_status === 'empty';
  const isIdentityUnverified = progress.library_status === 'identity_unverified';

  return (
    <div className="h-full overflow-y-auto" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '48px 24px' }}>
      <div style={{ width: '100%', maxWidth: 480 }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 32 }}>
          <div>
            <h1 style={{ fontSize: 20, fontWeight: 600 }}>{libraryName}</h1>
            <p style={{ fontSize: 12, color: '#a3a3a3', marginTop: 4 }}>
              {isCompleted ? 'Build complete' : isFailed ? 'Build failed' : isCancelled ? 'Build cancelled' : isQueued ? 'Queued' : 'Building'}
            </p>
          </div>
          <button
            onClick={() => navigate('/')}
            style={{ padding: 8, borderRadius: 6, background: 'none', border: 'none', cursor: 'pointer', color: '#a3a3a3' }}
            aria-label="Back to home"
          >
            <X size={18} />
          </button>
        </div>

        {/* Status banner */}
        {isFailed && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              padding: 16,
              backgroundColor: 'rgba(220,38,38,0.06)',
              border: '1px solid rgba(220,38,38,0.12)',
              borderRadius: 10,
              marginBottom: 24,
              display: 'flex',
              alignItems: 'center',
              gap: 12,
            }}
          >
            <AlertCircle size={16} style={{ color: '#dc2626', flexShrink: 0 }} />
            <span style={{ fontSize: 13, color: '#dc2626' }}>{progress.error || 'Build failed'}</span>
          </motion.div>
        )}

        {isCancelled && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              padding: 16,
              backgroundColor: 'rgba(163,163,163,0.06)',
              border: '1px solid rgba(163,163,163,0.12)',
              borderRadius: 10,
              marginBottom: 24,
              display: 'flex',
              alignItems: 'center',
              gap: 12,
            }}
          >
            <RotateCw size={16} style={{ color: '#a3a3a3', flexShrink: 0 }} />
            <span style={{ fontSize: 13, color: '#525252' }}>Build was cancelled</span>
          </motion.div>
        )}

        {isQueued && !isRunning && !isCompleted && !isFailed && !isCancelled && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              padding: 16,
              backgroundColor: 'rgba(163,163,163,0.06)',
              border: '1px solid rgba(163,163,163,0.12)',
              borderRadius: 10,
              marginBottom: 24,
              display: 'flex',
              alignItems: 'center',
              gap: 12,
            }}
          >
            <Clock size={16} style={{ color: '#737373', flexShrink: 0 }} />
            <span style={{ fontSize: 13, color: '#525252' }}>
              Waiting for the current build to finish...
            </span>
          </motion.div>
        )}

        {/* Progress bar */}
        {isRunning && (
          <div style={{ marginBottom: 24 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <span style={{ fontSize: 13, color: '#525252' }}>
                {progress.stage_label || 'Starting'}
              </span>
              {progress.items_total > 0 && (
                <span style={{ fontSize: 12, color: '#a3a3a3', fontVariantNumeric: 'tabular-nums' }}>
                  {progress.items_processed.toLocaleString()} / {progress.items_total.toLocaleString()}
                </span>
              )}
            </div>
            <div style={{ height: 6, backgroundColor: '#f5f5f5', borderRadius: 9999, overflow: 'hidden' }}>
              {progress.items_total > 0 ? (
                <motion.div
                  style={{ height: '100%', backgroundColor: '#0a0a0a', borderRadius: 9999 }}
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.3 }}
                />
              ) : (
                <motion.div
                  style={{ height: '100%', backgroundColor: '#0a0a0a', borderRadius: 9999, width: '30%' }}
                  animate={{ x: ['-100%', '400%'] }}
                  transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
                />
              )}
            </div>
          </div>
        )}

        {isCompleted && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            style={{
              padding: 20,
              backgroundColor: 'rgba(22,163,74,0.06)',
              border: '1px solid rgba(22,163,74,0.12)',
              borderRadius: 10,
              marginBottom: 24,
              display: 'flex',
              alignItems: 'center',
              gap: 12,
            }}
          >
            <CheckCircle2 size={18} style={{ color: '#16a34a', flexShrink: 0 }} />
            <span style={{ fontSize: 14, color: '#16a34a', fontWeight: 500 }}>Build complete</span>
          </motion.div>
        )}

        {/* Stage list */}
        {!isQueued && (
        <div style={{ marginBottom: 24 }}>
          <p style={{ fontSize: 10, color: '#a3a3a3', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 12 }}>
            Pipeline Stages
          </p>
          {ALL_STAGES.map((stage) => (
            <StageIndicator
              key={stage}
              stageName={stage}
              currentStage={progress.current_stage}
              stagesCompleted={stagesCompleted}
            />
          ))}
        </div>
        )}

        {/* Gandalf easter egg */}
        {isRunning && <GandalfFact stage={progress.current_stage || ''} />}

        {/* Actions */}
        <div style={{ display: 'flex', gap: 10, marginTop: 32 }}>
          {(isRunning || isQueued) && (
            <button
              onClick={() => cancelMut.mutate()}
              disabled={cancelMut.isPending}
              style={{
                padding: '10px 20px',
                fontSize: 13,
                borderRadius: 8,
                border: '1px solid #e5e5e5',
                backgroundColor: 'transparent',
                color: '#525252',
                cursor: 'pointer',
              }}
            >
              {cancelMut.isPending ? 'Cancelling...' : 'Cancel Build'}
            </button>
          )}
          {(isFailed || isCancelled) && (
            <button
              onClick={() => restartMut.mutate()}
              disabled={restartMut.isPending}
              style={{
                padding: '10px 20px',
                fontSize: 13,
                fontWeight: 500,
                borderRadius: 8,
                border: 'none',
                backgroundColor: '#0a0a0a',
                color: '#ffffff',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }}
            >
              {restartMut.isPending ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <>
                  <RotateCw size={14} />
                  Restart Build
                </>
              )}
            </button>
          )}
          {!isRunning && (
            <button
              onClick={() => navigate('/')}
              style={{
                padding: '10px 20px',
                fontSize: 13,
                borderRadius: 8,
                border: '1px solid #e5e5e5',
                backgroundColor: 'transparent',
                color: '#525252',
                cursor: 'pointer',
              }}
            >
              Back to Home
            </button>
          )}
          {isCompleted && (
            <button
              onClick={() => navigate(`/library/${encodeURIComponent(libraryName)}`)}
              style={{
                padding: '10px 20px',
                fontSize: 13,
                fontWeight: 500,
                borderRadius: 8,
                border: 'none',
                backgroundColor: '#0a0a0a',
                color: '#ffffff',
                cursor: 'pointer',
              }}
            >
              View Library
            </button>
          )}
          {isCompleted && isEmpty && (
            <span style={{ fontSize: 12, color: '#a3a3a3', alignSelf: 'center' }}>
              No images passed the configured filters.
            </span>
          )}
          {isCompleted && isIdentityUnverified && (
            <span style={{ fontSize: 12, color: '#b45309', alignSelf: 'center' }}>
              Identity could not be verified. Add manual seed images and rebuild.
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

import { useNavigate } from 'react-router-dom';
import { Trash2, Loader2, AlertCircle, RotateCw, CheckCircle2, Clock } from 'lucide-react';
import { motion } from 'framer-motion';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import type { Library } from '../types';
import { useState } from 'react';
import { ConfirmDialog } from './ConfirmDialog';

function StatusIndicator({ status, position }: { status: Library['status']; position?: number | null }) {
  if (status === 'queued') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <Clock size={12} style={{ color: '#a3a3a3' }} />
        <span style={{ fontSize: 10, color: '#737373', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Queued{position != null ? ` #${position}` : ''}
        </span>
      </div>
    );
  }
  if (status === 'building') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <Loader2 size={12} className="animate-spin" style={{ color: '#0a0a0a' }} />
        <span style={{ fontSize: 10, color: '#525252', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Building
        </span>
      </div>
    );
  }
  if (status === 'failed') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <AlertCircle size={12} style={{ color: '#dc2626' }} />
        <span style={{ fontSize: 10, color: '#dc2626', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Failed
        </span>
      </div>
    );
  }
  if (status === 'cancelled') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <RotateCw size={12} style={{ color: '#a3a3a3' }} />
        <span style={{ fontSize: 10, color: '#a3a3a3', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Cancelled
        </span>
      </div>
    );
  }
  if (status === 'empty') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <AlertCircle size={12} style={{ color: '#a3a3a3' }} />
        <span style={{ fontSize: 10, color: '#a3a3a3', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Empty
        </span>
      </div>
    );
  }
  if (status === 'identity_unverified') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <AlertCircle size={12} style={{ color: '#b45309' }} />
        <span style={{ fontSize: 10, color: '#b45309', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Identity Unverified
        </span>
      </div>
    );
  }
  if (status === 'ready') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <CheckCircle2 size={12} style={{ color: '#16a34a' }} />
        <span style={{ fontSize: 10, color: '#16a34a', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Ready
        </span>
      </div>
    );
  }
  return null;
}

function BuildProgressMini({ library }: { library: Library }) {
  const { build } = library;
  if (!build || (build.status !== 'running' && build.status !== 'pending')) return null;
  if (library.status === 'queued') return null;

  const pct = build.items_total > 0
    ? Math.round((build.items_processed / build.items_total) * 100)
    : 0;

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 10, color: '#525252', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '70%' }}>
          {build.stage_label || 'Starting'}
        </span>
        <span style={{ fontSize: 10, color: '#a3a3a3', fontVariantNumeric: 'tabular-nums' }}>
          {build.items_processed.toLocaleString()} / {build.items_total.toLocaleString()}
        </span>
      </div>
      <div style={{ height: 3, backgroundColor: '#ffffff', borderRadius: 9999, overflow: 'hidden' }}>
        <motion.div
          style={{
            height: '100%',
            backgroundColor: '#0a0a0a',
            borderRadius: 9999,
          }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>
    </div>
  );
}

export function LibraryCard({
  library,
  onRefresh,
}: {
  library: Library;
  onRefresh?: () => void;
}) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [hovered, setHovered] = useState(false);
  const [showConfirmDelete, setShowConfirmDelete] = useState(false);

  const isBuilding = library.status === 'building';
  const isQueued = library.status === 'queued';
  const isFailed = library.status === 'failed';
  const isCancelled = library.status === 'cancelled';
  const isEmpty = library.status === 'empty';

  const deleteMut = useMutation({
    mutationFn: () => api.deleteLibrary(library.name),
    onSuccess: () => {
      queryClient.setQueryData(['libraries'], (old: Library[] | undefined) =>
        old?.filter((l) => l.name !== library.name)
      );
      onRefresh?.();
    },
  });

  const restartMut = useMutation({
    mutationFn: () => api.startBuild(library.name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['libraries'] });
    },
  });

  const thumbnailUrl = library.thumbnail_hash
    ? `/api/libraries/${encodeURIComponent(library.name)}/images/${library.thumbnail_hash}/file`
    : null;

  const handleClick = () => {
    if (isBuilding || isQueued) {
      navigate(`/build/${encodeURIComponent(library.name)}`);
    } else if (isEmpty) {
      navigate(`/build/${encodeURIComponent(library.name)}`);
    } else {
      navigate(`/library/${encodeURIComponent(library.name)}/practice`);
    }
  };

  return (
    <motion.div
      whileHover={{ y: -2 }}
      transition={{ duration: 0.2 }}
      style={{
        backgroundColor: '#f5f5f5',
        border: '1px solid #e5e5e5',
        borderRadius: 12,
        overflow: 'hidden',
        cursor: 'pointer',
        transition: 'border-color 0.2s, box-shadow 0.2s',
        borderColor: hovered ? '#d4d4d4' : '#e5e5e5',
        boxShadow: hovered ? '0 4px 20px -8px rgba(0,0,0,0.06)' : 'none',
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={handleClick}
      role="button"
      tabIndex={0}
      aria-label={`Open library ${library.name}`}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleClick();
        }
      }}
    >
      {/* Thumbnail */}
      <div
        style={{
          aspectRatio: '16/10',
          backgroundColor: '#eeeeee',
          overflow: 'hidden',
          position: 'relative',
        }}
      >
        {isBuilding && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              backgroundColor: 'rgba(245,245,245,0.9)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 2,
            }}
          >
            <Loader2 size={24} className="animate-spin" style={{ color: '#0a0a0a', marginBottom: 8 }} />
            <span style={{ fontSize: 11, color: '#525252' }}>
              {library.build?.stage_label || 'Starting'}
            </span>
          </div>
        )}
        {isQueued && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              backgroundColor: 'rgba(245,245,245,0.9)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 2,
            }}
          >
            <Clock size={24} style={{ color: '#737373', marginBottom: 8 }} />
            <span style={{ fontSize: 11, color: '#737373' }}>Waiting in queue</span>
          </div>
        )}
        {isFailed && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              backgroundColor: 'rgba(245,245,245,0.9)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 2,
            }}
          >
            <AlertCircle size={24} style={{ color: '#dc2626', marginBottom: 8 }} />
            <span style={{ fontSize: 11, color: '#dc2626' }}>
              {library.build?.error || 'Build failed'}
            </span>
          </div>
        )}
        {isCancelled && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              backgroundColor: 'rgba(245,245,245,0.9)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 2,
            }}
          >
            <RotateCw size={24} style={{ color: '#a3a3a3', marginBottom: 8 }} />
            <span style={{ fontSize: 11, color: '#a3a3a3' }}>Build cancelled</span>
          </div>
        )}
        {isEmpty && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              backgroundColor: 'rgba(245,245,245,0.9)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 2,
            }}
          >
            <AlertCircle size={24} style={{ color: '#a3a3a3', marginBottom: 8 }} />
            <span style={{ fontSize: 11, color: '#a3a3a3' }}>No valid images found</span>
          </div>
        )}
        {thumbnailUrl ? (
          <img
            src={thumbnailUrl}
            alt=""
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            loading="lazy"
          />
        ) : (
          <div
            style={{
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <span
              style={{
                fontSize: 48,
                fontWeight: 900,
                color: '#e5e5e5',
                userSelect: 'none',
              }}
            >
              {library.name.charAt(0)}
            </span>
          </div>
        )}
      </div>

      {/* Info */}
      <div style={{ padding: '14px 16px 16px' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            marginBottom: 12,
          }}
        >
          <div style={{ minWidth: 0, flex: 1 }}>
            <h3 style={{ fontWeight: 600, fontSize: 13, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {library.name}
            </h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 2 }}>
              <StatusIndicator status={library.status} position={library.build?.queue_position} />
              {!isBuilding && !isQueued && (
                <p style={{ fontSize: 11, color: '#a3a3a3' }}>
                  {library.image_count.toLocaleString()} references
                </p>
              )}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 4 }}>
            {(isFailed || isCancelled || isEmpty) && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  restartMut.mutate();
                }}
                style={{
                  padding: 4,
                  borderRadius: 4,
                  opacity: hovered ? 1 : 0,
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  color: '#525252',
                  transition: 'opacity 0.2s',
                  display: 'flex',
                  alignItems: 'center',
                }}
                title="Restart build"
                aria-label={`Restart build for ${library.name}`}
              >
                <RotateCw size={12} />
              </button>
            )}
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowConfirmDelete(true);
              }}
              style={{
                padding: 4,
                borderRadius: 4,
                opacity: hovered ? 1 : 0,
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: '#a3a3a3',
                transition: 'opacity 0.2s, color 0.2s',
              }}
              title="Delete library"
              aria-label={`Delete library ${library.name}`}
              onMouseEnter={(e) => { e.currentTarget.style.color = '#dc2626'; }}
              onMouseLeave={(e) => { e.currentTarget.style.color = '#a3a3a3'; }}
            >
              <Trash2 size={12} />
            </button>
          </div>
        </div>

        {/* Build progress */}
        <BuildProgressMini library={library} />

        {/* Quality/Coverage bars */}
        {!isBuilding && library.image_count > 0 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: isBuilding ? 0 : 0 }}>
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 10, color: '#a3a3a3', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Q
                </span>
                <span style={{ fontSize: 10, color: '#525252', fontVariantNumeric: 'tabular-nums' }}>
                  {Math.round(library.quality_score * 100)}
                </span>
              </div>
              <div style={{ height: 3, backgroundColor: '#ffffff', borderRadius: 9999, overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    borderRadius: 9999,
                    backgroundColor: 'rgba(10,10,10,0.3)',
                    width: `${Math.round(library.quality_score * 100)}%`,
                  }}
                />
              </div>
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 10, color: '#a3a3a3', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  C
                </span>
                <span style={{ fontSize: 10, color: '#525252', fontVariantNumeric: 'tabular-nums' }}>
                  {Math.round(library.coverage_score * 100)}
                </span>
              </div>
              <div style={{ height: 3, backgroundColor: '#ffffff', borderRadius: 9999, overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    borderRadius: 9999,
                    backgroundColor: 'rgba(10,10,10,0.3)',
                    width: `${Math.round(library.coverage_score * 100)}%`,
                  }}
                />
              </div>
            </div>
          </div>
        )}
      </div>
      {showConfirmDelete && (
        <ConfirmDialog
          title="Delete library"
          message={`Delete "${library.name}" and all its images? This cannot be undone.`}
          confirmLabel="Delete"
          onConfirm={() => {
            setShowConfirmDelete(false);
            deleteMut.mutate();
          }}
          onCancel={() => setShowConfirmDelete(false)}
        />
      )}
    </motion.div>
  );
}

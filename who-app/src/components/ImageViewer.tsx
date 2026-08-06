import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQueryClient } from '@tanstack/react-query';
import {
  X, ZoomIn, ZoomOut, RotateCcw, Info, Keyboard,
  ChevronLeft, ChevronRight, Trash2,
} from 'lucide-react';
import { useAppStore } from '../store';
import { api } from '../api/client';

export function ImageViewer() {
  const {
    viewerOpen, selectedImage, viewerLibraryName,
    viewerImages, viewerIndex, navigateViewer, removeViewerImage, closeViewer,
  } = useAppStore();
  const queryClient = useQueryClient();
  const [zoom, setZoom] = useState(1);
  const [showInfo, setShowInfo] = useState(false);
  const [showHelp, setShowHelp] = useState(false);

  useEffect(() => {
    setZoom(1);
  }, [selectedImage?.content_hash]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom((z) => Math.max(0.2, Math.min(5, z + delta)));
  }, []);

  useEffect(() => {
    if (!viewerOpen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeViewer();
      if (e.key === '+' || e.key === '=') setZoom((z) => Math.min(5, z + 0.2));
      if (e.key === '-') setZoom((z) => Math.max(0.2, z - 0.2));
      if (e.key === '0') setZoom(1);
      if (e.key === 'i') setShowInfo((s) => !s);
      if (e.key === '?') setShowHelp((s) => !s);
      if (e.key === 'ArrowLeft') navigateViewer(-1);
      if (e.key === 'ArrowRight') navigateViewer(1);
      if (e.key === 'Delete' && selectedImage) {
        api.deleteImage(viewerLibraryName, selectedImage.content_hash).then(() => {
          removeViewerImage(selectedImage.content_hash);
          queryClient.invalidateQueries({ queryKey: ['images', viewerLibraryName] });
        });
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [viewerOpen, closeViewer, navigateViewer, selectedImage, viewerLibraryName, removeViewerImage, queryClient]);

  if (!viewerOpen || !selectedImage) return null;

  const q = selectedImage.quality;
  const f = selectedImage.face;
  const c = selectedImage.classification;
  const s = selectedImage.safety;
  const hasPrev = viewerIndex > 0;
  const hasNext = viewerIndex < viewerImages.length - 1;
  const total = viewerImages.length;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 bg-black/95 flex items-center justify-center"
        onWheel={handleWheel}
        onClick={(e) => { if (e.target === e.currentTarget) closeViewer(); }}
        role="dialog"
        aria-modal="true"
        aria-label="Image viewer"
      >
        {/* Top toolbar */}
        <div className="absolute top-4 right-4 z-10 flex items-center gap-1">
          <button
            onClick={() => setShowHelp(!showHelp)}
            className={`p-2 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-white/30 ${
              showHelp ? 'bg-white/20 text-white' : 'bg-white/10 text-white/60 hover:text-white'
            }`}
            aria-label="Keyboard shortcuts"
          >
            <Keyboard size={18} />
          </button>
          <button
            onClick={() => setShowInfo(!showInfo)}
            className={`p-2 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-white/30 ${
              showInfo ? 'bg-white/20 text-white' : 'bg-white/10 text-white/60 hover:text-white'
            }`}
            aria-label="Image info"
          >
            <Info size={18} />
          </button>
          <button
            onClick={() => setZoom((z) => Math.max(0.2, z - 0.3))}
            className="p-2 rounded-lg bg-white/10 text-white/60 hover:text-white transition-colors focus:outline-none focus:ring-2 focus:ring-white/30"
            aria-label="Zoom out"
          >
            <ZoomOut size={18} />
          </button>
          <span className="text-xs text-white/50 tabular-nums w-10 text-center">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={() => setZoom((z) => Math.min(5, z + 0.3))}
            className="p-2 rounded-lg bg-white/10 text-white/60 hover:text-white transition-colors focus:outline-none focus:ring-2 focus:ring-white/30"
            aria-label="Zoom in"
          >
            <ZoomIn size={18} />
          </button>
          <button
            onClick={() => setZoom(1)}
            className="p-2 rounded-lg bg-white/10 text-white/60 hover:text-white transition-colors focus:outline-none focus:ring-2 focus:ring-white/30"
            aria-label="Reset zoom"
          >
            <RotateCcw size={18} />
          </button>
          <div className="w-px h-4 bg-white/20 mx-1" />
          <button
            onClick={closeViewer}
            className="p-2 rounded-lg bg-white/10 text-white/60 hover:text-white transition-colors focus:outline-none focus:ring-2 focus:ring-white/30"
            aria-label="Close viewer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Counter */}
        {total > 1 && (
          <div className="absolute top-4 left-4 z-10 text-xs text-white/40 tabular-nums">
            {viewerIndex + 1} / {total}
          </div>
        )}

        {/* Left arrow */}
        {hasPrev && (
          <button
            onClick={() => navigateViewer(-1)}
            className="absolute left-4 top-1/2 -translate-y-1/2 z-10 p-3 rounded-full bg-white/10 text-white/50 hover:text-white hover:bg-white/20 transition-colors focus:outline-none focus:ring-2 focus:ring-white/30"
            aria-label="Previous image"
          >
            <ChevronLeft size={24} />
          </button>
        )}

        {/* Right arrow */}
        {hasNext && (
          <button
            onClick={() => navigateViewer(1)}
            className="absolute right-4 top-1/2 -translate-y-1/2 z-10 p-3 rounded-full bg-white/10 text-white/50 hover:text-white hover:bg-white/20 transition-colors focus:outline-none focus:ring-2 focus:ring-white/30"
            aria-label="Next image"
          >
            <ChevronRight size={24} />
          </button>
        )}

        {/* Image */}
        <div className="relative flex items-center justify-center w-full h-full overflow-hidden pointer-events-none">
          <AnimatePresence mode="wait">
            <motion.img
              key={selectedImage.content_hash}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.15 }}
              src={api.getImageUrl(viewerLibraryName, selectedImage.content_hash)}
              alt=""
              className="max-w-full max-h-full object-contain select-none pointer-events-auto"
              style={{
                transform: `scale(${zoom})`,
                transition: 'transform 0.1s ease-out',
              }}
              draggable={false}
            />
          </AnimatePresence>
        </div>

        {/* Help panel */}
        <AnimatePresence>
          {showHelp && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="absolute top-14 left-1/2 -translate-x-1/2 bg-black/80 backdrop-blur-md rounded-xl p-4 text-xs text-white/70 space-y-1.5"
            >
              {[
                ['← →', 'Previous / Next'],
                ['+ -', 'Zoom in / out'],
                ['0', 'Reset zoom'],
                ['I', 'Toggle info'],
                ['Del', 'Delete image'],
                ['Esc', 'Close'],
              ].map(([key, desc]) => (
                <div key={key} className="flex items-center gap-3">
                  <kbd className="px-1.5 py-0.5 bg-white/10 rounded text-white/50 font-mono w-12 text-center">
                    {key}
                  </kbd>
                  <span>{desc}</span>
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Info panel */}
        <AnimatePresence>
          {showInfo && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="absolute right-4 top-14 w-64 bg-black/80 backdrop-blur-md rounded-xl p-4 space-y-3 text-sm"
            >
              {q && (
                <div>
                  <h4 className="text-white/40 text-xs uppercase tracking-wider mb-2">
                    Quality
                  </h4>
                  <div className="grid grid-cols-2 gap-1">
                    {[
                      ['Overall', q.final_score],
                      ['Sharpness', q.sharpness_score],
                      ['Lighting', q.lighting_score],
                      ['Resolution', q.resolution_score],
                    ].map(([label, score]) => (
                      <div key={label} className="flex justify-between">
                        <span className="text-white/60">{label}</span>
                        <span className="text-white/80 tabular-nums">
                          {Math.round((score as number) * 100)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {f && (
                <div>
                  <h4 className="text-white/40 text-xs uppercase tracking-wider mb-2">
                    Face
                  </h4>
                  <div className="grid grid-cols-2 gap-1">
                    <div className="flex justify-between">
                      <span className="text-white/60">Yaw</span>
                      <span className="text-white/80 tabular-nums">
                        {Math.round(f.yaw)}°
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white/60">Pitch</span>
                      <span className="text-white/80 tabular-nums">
                        {Math.round(f.pitch)}°
                      </span>
                    </div>
                  </div>
                </div>
              )}
              {c && (
                <div>
                  <h4 className="text-white/40 text-xs uppercase tracking-wider mb-2">
                    Classification
                  </h4>
                  <div className="space-y-1">
                    {c.horizontal_pose && (
                      <div className="flex justify-between">
                        <span className="text-white/60">Angle</span>
                        <span className="text-white/80">{c.horizontal_pose}</span>
                      </div>
                    )}
                    {c.vertical_pose && c.vertical_pose !== 'neutral' && (
                      <div className="flex justify-between">
                        <span className="text-white/60">Vertical Pose</span>
                        <span className="text-white/80">{c.vertical_pose}</span>
                      </div>
                    )}
                    {c.expression && (
                      <div className="flex justify-between">
                        <span className="text-white/60">Expression</span>
                        <span className="text-white/80">{c.expression}</span>
                      </div>
                    )}
                    {c.lighting && (
                      <div className="flex justify-between">
                        <span className="text-white/60">Lighting</span>
                        <span className="text-white/80">{c.lighting}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
              {s && (
                <div>
                  <h4 className="text-white/40 text-xs uppercase tracking-wider mb-2">
                    Safety
                  </h4>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-white/60">Real photo</span>
                      <span className={`tabular-nums ${s.real_photo_score >= 0.7 ? 'text-green-400' : s.real_photo_score >= 0.4 ? 'text-yellow-400' : 'text-red-400'}`}>
                        {Math.round(s.real_photo_score * 100)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white/60">Source trust</span>
                      <span className="text-white/80 tabular-nums">
                        {Math.round(s.source_trust_score * 100)}%
                      </span>
                    </div>
                    {s.is_ai_generated && (
                      <div className="flex justify-between">
                        <span className="text-white/60">AI generated</span>
                        <span className="text-yellow-400">Yes</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
              <div className="pt-2 border-t border-white/10">
                <div className="flex justify-between">
                  <span className="text-white/40">Size</span>
                  <span className="text-white/60">
                    {selectedImage.width} × {selectedImage.height}
                  </span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </AnimatePresence>
  );
}

import { useEffect } from 'react';
import { X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export function ConfirmDialog({
  title,
  message,
  confirmLabel = 'Delete',
  onConfirm,
  onCancel,
}: {
  title: string;
  message: string;
  confirmLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onCancel]);

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
        onClick={onCancel}
        role="dialog"
        aria-modal="true"
        aria-label={title}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
          className="bg-bg-card rounded-xl shadow-xl w-full max-w-sm mx-4 overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between px-6 py-4 border-b border-border">
            <h2 className="text-sm font-semibold text-text-secondary">{title}</h2>
            <button
              onClick={onCancel}
              className="p-1.5 rounded-md text-text-muted hover:text-text hover:bg-bg-hover transition-colors"
              aria-label="Close dialog"
            >
              <X size={16} />
            </button>
          </div>

          <div className="p-6">
            <p className="text-sm text-text-secondary leading-relaxed">
              {message}
            </p>
          </div>

          <div className="flex justify-end gap-2 px-6 py-4 border-t border-border bg-bg-card">
            <button
              onClick={onCancel}
              className="px-5 py-2 text-sm text-text-secondary hover:text-text rounded-lg border border-border hover:bg-bg-elevated transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={onConfirm}
              className="px-5 py-2 text-sm font-medium text-white bg-danger rounded-lg hover:opacity-90 transition-opacity"
            >
              {confirmLabel}
            </button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

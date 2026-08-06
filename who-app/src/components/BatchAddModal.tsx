import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Plus, AlertCircle } from 'lucide-react';
import { api } from '../api/client';
import { useQueryClient } from '@tanstack/react-query';

function parseNames(raw: string): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const line of raw.split('\n')) {
    let name = line.trim();
    name = name.replace(/^\d+[\.\)\-\s]+/, '');
    name = name.replace(/^[-•*]\s+/, '');
    name = name.trim();
    if (!name) continue;
    const key = name.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(name);
  }
  return result;
}

interface BatchAddModalProps {
  open: boolean;
  onClose: () => void;
}

export default function BatchAddModal({ open, onClose }: BatchAddModalProps) {
  const [raw, setRaw] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    added: string[];
    already_exists: Array<{ name: string; status: string }>;
    queued: number;
  } | null>(null);
  const [error, setError] = useState('');
  const queryClient = useQueryClient();

  const names = parseNames(raw);

  const handleSubmit = useCallback(async () => {
    if (names.length === 0) return;
    setLoading(true);
    setError('');
    try {
      const res = await api.batchEnqueue(names);
      setResult(res);
      queryClient.invalidateQueries({ queryKey: ['libraries'] });
      queryClient.invalidateQueries({ queryKey: ['buildQueue'] });
    } catch (e: any) {
      setError(e.message || 'Failed to enqueue');
    } finally {
      setLoading(false);
    }
  }, [names, queryClient]);

  const handleClose = useCallback(() => {
    setRaw('');
    setResult(null);
    setError('');
    onClose();
  }, [onClose]);

  if (!open) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
        onClick={handleClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="bg-bg-card rounded-xl shadow-xl w-full max-w-lg mx-4 overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between px-6 py-4 border-b border-border">
            <h2 className="text-sm font-semibold text-text-secondary">Add people</h2>
            <button onClick={handleClose} className="text-text-muted hover:text-text transition-colors">
              <X size={18} />
            </button>
          </div>

          <div className="p-6">
            {!result ? (
              <>
                <p className="text-sm text-text-muted mb-3">Paste one name per line</p>
                <textarea
                  value={raw}
                  onChange={(e) => setRaw(e.target.value)}
                  placeholder={'Brad Pitt\nLeonardo DiCaprio\nChristian Bale'}
                  className="w-full h-48 px-3 py-2 text-sm border border-border rounded-lg resize-none focus:outline-none focus:ring-1 focus:ring-border-hover font-mono"
                />
                {names.length > 0 && (
                  <p className="text-xs text-text-muted mt-2">{names.length} {names.length === 1 ? 'name' : 'names'} detected</p>
                )}
                {error && (
                  <div className="flex items-center gap-2 mt-3 text-sm text-danger">
                    <AlertCircle size={14} />
                    {error}
                  </div>
                )}
              </>
            ) : (
              <div className="space-y-4">
                <div className="text-sm">
                  <p className="text-text font-medium">
                    {result.added.length} {result.added.length === 1 ? 'library' : 'libraries'} added to queue
                  </p>
                  {result.already_exists.length > 0 && (
                    <p className="text-text-muted mt-1">
                      {result.already_exists.length} already {result.already_exists.length === 1 ? 'exists' : 'exist'}
                    </p>
                  )}
                </div>
                {result.added.length > 0 && (
                  <div className="text-sm">
                    <p className="text-xs text-text-muted uppercase tracking-wide mb-2">Added</p>
                    <div className="space-y-1">
                      {result.added.map((n) => (
                        <p key={n} className="text-text-secondary flex items-center gap-1.5">
                          <Plus size={12} className="text-success shrink-0" /> {n}
                        </p>
                      ))}
                    </div>
                  </div>
                )}
                {result.already_exists.length > 0 && (
                  <div className="text-sm">
                    <p className="text-xs text-text-muted uppercase tracking-wide mb-2">Already exists</p>
                    <div className="space-y-1">
                      {result.already_exists.map((e) => (
                        <p key={e.name} className="text-text-muted">
                          {e.name} — {e.status}
                        </p>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="flex justify-end gap-2 px-6 py-4 border-t border-border bg-bg-card">
            <button
              onClick={handleClose}
              className="px-4 py-2 text-sm text-text-secondary hover:text-text rounded-lg transition-colors"
            >
              {result ? 'Close' : 'Cancel'}
            </button>
            {!result && (
              <button
                onClick={handleSubmit}
                disabled={names.length === 0 || loading}
                className="px-5 py-2 text-sm bg-accent text-white rounded-lg hover:opacity-90 disabled:opacity-40 transition-opacity"
              >
                {loading ? 'Adding...' : `Add ${names.length} ${names.length === 1 ? 'person' : 'people'}`}
              </button>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

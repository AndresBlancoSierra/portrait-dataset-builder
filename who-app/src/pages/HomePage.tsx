import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Shuffle, Book as BookIcon, BookOpen, Users, Loader2, X } from 'lucide-react';
import { motion, useAnimation } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { LibraryCard } from '../components/LibraryCard';
import { LibraryCardSkeleton } from '../components/Skeletons';
import BatchAddModal from '../components/BatchAddModal';
import QueuePanel from '../components/QueuePanel';
import type { Book } from '../types';

function WhoLogo() {
  const controls = useAnimation();

  useEffect(() => {
    controls.start({
      letterSpacing: ['0.12em', '-0.02em'],
      transition: { duration: 0.8, ease: [0.22, 1, 0.36, 1] },
    });
  }, [controls]);

  return (
    <motion.h1
      animate={controls}
      initial={{ letterSpacing: '0.12em' }}
      className="select-none"
      style={{ fontSize: 'clamp(3rem, 8vw, 5.5rem)', fontWeight: 900, lineHeight: 1 }}
    >
      <span>WH</span>
      <span>O</span>
      <motion.span
        className="inline-block"
        animate={{ y: [0, -6, 0, -2, 0] }}
        transition={{
          duration: 1,
          repeat: Infinity,
          repeatDelay: 5,
          ease: 'easeInOut',
        }}
      >
        ?
      </motion.span>
    </motion.h1>
  );
}

export default function HomePage() {
  const [query, setQuery] = useState('');
  const [batchOpen, setBatchOpen] = useState(false);
  const [bookSelectOpen, setBookSelectOpen] = useState(false);
  const navigate = useNavigate();

  const { data: libraries = [], refetch, isLoading } = useQuery({
    queryKey: ['libraries'],
    queryFn: api.listLibraries,
  });

  const { data: books = [] } = useQuery({
    queryKey: ['books'],
    queryFn: api.listBooks,
    enabled: bookSelectOpen,
  });

  const filtered = query.trim()
    ? libraries.filter((l) =>
        l.name.toLowerCase().includes(query.trim().toLowerCase())
      )
    : libraries;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const name = query.trim();
    if (!name) return;

    const match = filtered[0];
    if (match) {
      if (match.status === 'building' || match.status === 'queued') {
        navigate(`/build/${encodeURIComponent(match.name)}`);
      } else {
        navigate(`/library/${encodeURIComponent(match.name)}/practice`);
      }
    }
  };

  const hasLibraries = libraries.length > 0;

  return (
    <div className="h-full overflow-y-auto">
      {/* Hero section */}
      <section
        className="flex flex-col items-center justify-center"
        style={{ minHeight: hasLibraries ? '50vh' : '85vh', padding: '40px 24px' }}
      >
        <div className="flex flex-col items-center" style={{ width: '100%', maxWidth: 520 }}>
          {/* Global Random Practice */}
          {hasLibraries && (
            <div className="flex gap-2" style={{ alignSelf: 'flex-end', marginBottom: 48 }}>
              <button
                onClick={() => navigate('/practice/random')}
                className="p-2.5 rounded-lg text-text-muted hover:text-text hover:bg-bg-elevated transition-colors"
                aria-label="Random practice"
                title="Random practice across all libraries"
              >
                <Shuffle size={18} />
              </button>
              <button
                onClick={() => setBookSelectOpen(true)}
                className="p-2.5 rounded-lg text-text-muted hover:text-text hover:bg-bg-elevated transition-colors"
                aria-label="Book practice"
                title="Practice with drawing book pages"
              >
                <BookIcon size={18} />
              </button>
            </div>
          )}

          {/* Logo */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          >
            <WhoLogo />
          </motion.div>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.5 }}
            className="text-text-muted text-sm tracking-wide"
            style={{ fontWeight: 500, marginTop: 16, marginBottom: 40 }}
          >
            Find. Study. Draw.
          </motion.p>

          {/* Search bar */}
          <motion.form
            onSubmit={handleSubmit}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
            style={{ width: '100%' }}
          >
            <div style={{ position: 'relative' }}>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search any public person..."
                style={{
                  width: '100%',
                  backgroundColor: '#f5f5f5',
                  border: '1px solid #e5e5e5',
                  borderRadius: 9999,
                  padding: '16px 60px 16px 20px',
                  fontSize: 15,
                  color: '#0a0a0a',
                  outline: 'none',
                }}
                aria-label="Search for a person"
              />
              <button
                type="submit"
                style={{
                  position: 'absolute',
                  right: 8,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  width: 40,
                  height: 40,
                  borderRadius: '50%',
                  backgroundColor: 'transparent',
                  color: '#a3a3a3',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: 'none',
                  cursor: 'pointer',
                }}
                aria-label="Search"
              >
                <Search size={20} />
              </button>
            </div>
          </motion.form>

          {/* Add people */}
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.3 }}
            onClick={() => setBatchOpen(true)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              marginTop: 20,
              padding: '8px 16px',
              borderRadius: 9999,
              border: '1px solid #e5e5e5',
              fontSize: 12,
              color: '#525252',
              backgroundColor: 'transparent',
              cursor: 'pointer',
            }}
          >
            <Users size={13} />
            Add people
          </motion.button>
        </div>
      </section>

      {/* Libraries section */}
      {hasLibraries && (
        <section style={{ padding: '0 24px 48px', maxWidth: 1024, margin: '0 auto' }}>
          <h2
            style={{
              fontSize: 11,
              fontWeight: 500,
              color: '#a3a3a3',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
              marginBottom: 20,
              textAlign: 'center',
            }}
          >
            {query.trim() ? `${filtered.length} result${filtered.length !== 1 ? 's' : ''}` : 'Libraries'}
          </h2>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
              gap: 20,
            }}
          >
            {filtered.map((lib) => (
              <motion.div
                key={lib.name}
                layout
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.25 }}
              >
                <LibraryCard library={lib} onRefresh={refetch} />
              </motion.div>
            ))}
          </div>
        </section>
      )}

      {isLoading && (
        <section style={{ padding: '0 24px 48px', maxWidth: 1024, margin: '0 auto' }}>
          <h2
            style={{
              fontSize: 11,
              fontWeight: 500,
              color: '#a3a3a3',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
              marginBottom: 20,
              textAlign: 'center',
            }}
          >
            Libraries
          </h2>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
              gap: 20,
            }}
          >
            {Array.from({ length: 3 }).map((_, i) => (
              <LibraryCardSkeleton key={i} />
            ))}
          </div>
        </section>
      )}

      {/* Queue panel */}
      {hasLibraries && (
        <section style={{ padding: '0 24px 48px', maxWidth: 1024, margin: '0 auto' }}>
          <QueuePanel />
        </section>
      )}

      <BatchAddModal open={batchOpen} onClose={() => setBatchOpen(false)} />

      {bookSelectOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="bg-bg-card rounded-xl shadow-xl w-full max-w-lg mx-4 max-h-[80vh] flex flex-col overflow-hidden">
            <div className="flex items-center justify-between px-6 py-4 border-b border-border shrink-0">
              <h2 className="text-sm font-semibold text-text-secondary">Drawing Books</h2>
              <button
                onClick={() => setBookSelectOpen(false)}
                className="text-text-muted hover:text-text transition-colors"
              >
                <X size={18} />
              </button>
            </div>
            <div className="overflow-y-auto p-6">
              <button
                onClick={() => { setBookSelectOpen(false); navigate('/practice/books'); }}
                className="w-full text-left px-4 py-3 mx-2 my-1 rounded-xl hover:bg-bg-elevated transition-colors flex items-center gap-3"
              >
                <div className="w-10 h-10 rounded-lg bg-bg-card flex items-center justify-center text-text-muted">
                  <BookOpen size={18} />
                </div>
                <div>
                  <div className="text-sm font-medium text-text">All Books</div>
                  <div className="text-xs text-text-muted">{books.length} books · random pages from all</div>
                </div>
              </button>
              <div className="mx-6 my-3 text-[10px] font-semibold text-text-muted uppercase tracking-wider">Categories</div>
              {['anatomy', 'figure-drawing'].map((cat) => {
                const catBooks = books.filter((b: Book) => b.category === cat);
                if (catBooks.length === 0) return null;
                return (
                  <div key={cat}>
                    <div className="px-6 py-1 text-[11px] font-medium text-text-muted capitalize">{cat}</div>
                    {catBooks.map((book: Book) => (
                      <button
                        key={book.slug}
                        onClick={() => { setBookSelectOpen(false); navigate(`/practice/books/${book.slug}`); }}
                        className="w-full text-left px-4 py-2.5 mx-2 my-0.5 rounded-xl hover:bg-bg-elevated transition-colors flex items-center gap-3"
                      >
                        <div className="w-8 h-8 rounded-lg bg-bg-card flex items-center justify-center text-text-muted">
                          <BookIcon size={16} />
                        </div>
                        <div>
                          <div className="text-sm text-text">{book.title}</div>
                          <div className="text-xs text-text-muted">{book.page_count} pages</div>
                        </div>
                      </button>
                    ))}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

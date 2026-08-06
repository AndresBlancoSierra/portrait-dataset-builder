import { useParams, useNavigate, Navigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, Shuffle, Eye, Grid3X3, BarChart3, Layers, CheckCircle, AlertCircle } from 'lucide-react';
import { api } from '../api/client';
import { Gallery } from '../components/Gallery';
import { FiltersPanel } from '../components/FiltersPanel';
import { CoverageHeatmap } from '../components/CoverageHeatmap';
import { ReviewPanel } from '../components/ReviewPanel';
import { StatsDashboard } from '../components/StatsDashboard';
import { useAppStore } from '../store';
import { useState } from 'react';

type SidebarTab = 'filters' | 'coverage' | 'review' | 'stats';

const TABS: { key: SidebarTab; icon: typeof Grid3X3; label: string }[] = [
  { key: 'filters', icon: Grid3X3, label: 'Filters' },
  { key: 'coverage', icon: Layers, label: 'Coverage' },
  { key: 'review', icon: CheckCircle, label: 'Review' },
  { key: 'stats', icon: BarChart3, label: 'Stats' },
];

export default function LibraryPage() {
  const { name } = useParams<{ name: string }>();
  const navigate = useNavigate();
  const decodedName = decodeURIComponent(name || '');
  const { sidebarOpen, toggleSidebar } = useAppStore();
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [sortBy, setSortBy] = useState('quality');
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>('filters');

  const { data: library } = useQuery({
    queryKey: ['library', decodedName],
    queryFn: () => api.getLibrary(decodedName),
    enabled: !!decodedName,
  });

  const { data: images = [], isLoading, error: imagesError } = useQuery({
    queryKey: ['images', decodedName, filters, sortBy],
    queryFn: () => api.getImages(decodedName, { ...filters, sort: sortBy }),
    enabled: !!decodedName,
  });

  if (!decodedName) return null;

  // Redirect building/queued libraries to build progress page
  if (library?.status === 'building' || library?.status === 'queued') {
    return <Navigate to={`/build/${encodeURIComponent(decodedName)}`} replace />;
  }

  // Empty state
  if (library?.status === 'empty') {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4">
        <AlertCircle size={24} style={{ color: '#a3a3a3' }} />
        <p style={{ fontSize: 14, color: '#525252', fontWeight: 500 }}>No valid images found</p>
        <p style={{ fontSize: 12, color: '#a3a3a3' }}>
          The pipeline completed but no images passed the configured filters.
        </p>
        <button
          onClick={() => navigate('/')}
          style={{
            padding: '10px 20px', fontSize: 13, borderRadius: 8,
            border: '1px solid #e5e5e5', backgroundColor: 'transparent',
            color: '#525252', cursor: 'pointer', marginTop: 8,
          }}
        >
          Back to Home
        </button>
      </div>
    );
  }

  // Identity unverified state
  if (library?.status === 'identity_unverified') {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4">
        <AlertCircle size={24} style={{ color: '#b45309' }} />
        <p style={{ fontSize: 14, color: '#525252', fontWeight: 500 }}>Identity could not be verified</p>
        <p style={{ fontSize: 12, color: '#a3a3a3' }}>
          The pipeline could not confirm this is the same person. Add manual seed images and rebuild.
        </p>
        <button
          onClick={() => navigate('/')}
          style={{
            padding: '10px 20px', fontSize: 13, borderRadius: 8,
            border: '1px solid #e5e5e5', backgroundColor: 'transparent',
            color: '#525252', cursor: 'pointer', marginTop: 8,
          }}
        >
          Back to Home
        </button>
      </div>
    );
  }

  // Failed state
  if (library?.status === 'failed') {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4">
        <AlertCircle size={24} style={{ color: '#dc2626' }} />
        <p style={{ fontSize: 14, color: '#525252', fontWeight: 500 }}>Build failed</p>
        <p style={{ fontSize: 12, color: '#a3a3a3' }}>{library.build?.error || 'An error occurred during the build.'}</p>
        <button
          onClick={() => navigate('/')}
          style={{
            padding: '10px 20px', fontSize: 13, borderRadius: 8,
            border: '1px solid #e5e5e5', backgroundColor: 'transparent',
            color: '#525252', cursor: 'pointer', marginTop: 8,
          }}
        >
          Back to Home
        </button>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <header className="flex items-center gap-4 px-6 py-4 border-b border-border shrink-0">
        <button
          onClick={() => navigate('/')}
          className="p-2 rounded-lg hover:bg-bg-hover transition-colors"
        >
          <ArrowLeft size={18} />
        </button>

        <div className="flex-1 min-w-0">
          <h1 className="text-lg font-medium truncate">{decodedName}</h1>
          {library && (
            <p className="text-sm text-text-secondary">
              {library.image_count.toLocaleString()} references
            </p>
          )}
        </div>

        <div className="flex items-center gap-1">
          {[
            { key: 'quality', icon: BarChart3, label: 'Quality' },
            { key: 'random', icon: Shuffle, label: 'Random' },
          ].map(({ key, icon: Icon, label }) => (
            <button
              key={key}
              onClick={() => setSortBy(key)}
              className={`p-2 rounded-lg transition-colors ${
                sortBy === key ? 'bg-bg-elevated text-text' : 'text-text-muted hover:text-text'
              }`}
              title={label}
            >
              <Icon size={16} />
            </button>
          ))}
          <div className="w-px h-4 bg-border mx-1" />
          <button
            onClick={() => navigate(`/library/${encodeURIComponent(decodedName)}/practice`)}
            className="p-2 rounded-lg text-text-muted hover:text-text transition-colors"
            title="Practice"
          >
            <Eye size={16} />
          </button>
          <button
            onClick={toggleSidebar}
            className={`p-2 rounded-lg transition-colors ${
              sidebarOpen ? 'bg-bg-elevated text-text' : 'text-text-muted hover:text-text'
            }`}
            title="Sidebar"
          >
            <Grid3X3 size={16} />
          </button>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 overflow-hidden">
          <Gallery
            images={images}
            isLoading={isLoading}
            error={imagesError}
            libraryName={decodedName}
          />
        </div>
        {sidebarOpen && (
          <div className="w-80 border-l border-border flex flex-col shrink-0">
            <div className="flex border-b border-border">
              {TABS.map(({ key, icon: Icon, label }) => (
                <button
                  key={key}
                  onClick={() => setSidebarTab(key)}
                  className={`flex-1 flex items-center justify-center gap-1.5 py-2.5 text-xs transition-colors ${
                    sidebarTab === key
                      ? 'text-text border-b-2 border-text'
                      : 'text-text-muted hover:text-text'
                  }`}
                >
                  <Icon size={14} />
                  <span className="hidden xl:inline">{label}</span>
                </button>
              ))}
            </div>

            <div className="flex-1 overflow-y-auto p-4">
              {sidebarTab === 'filters' && (
                <FiltersPanel filters={filters} onChange={setFilters} />
              )}
              {sidebarTab === 'coverage' && (
                <CoverageHeatmap libraryName={decodedName} />
              )}
              {sidebarTab === 'review' && (
                <ReviewPanel libraryName={decodedName} />
              )}
              {sidebarTab === 'stats' && (
                <StatsDashboard libraryName={decodedName} />
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import Viewer from '../components/Viewer';
import type { ViewerItem } from '../components/Viewer';
import type { GlobalImage } from '../types';

export default function GlobalPracticePage() {
  const navigate = useNavigate();

  const { data: images = [], isLoading } = useQuery({
    queryKey: ['practice', 'random-global'],
    queryFn: () => api.getRandomGlobal(100),
  });

  if (isLoading) {
    return (
      <div className="fixed inset-0 z-50 bg-bg flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-5 h-5 border-2 border-white/20 border-t-white/70 rounded-full animate-spin" />
          <span className="text-xs text-white/40 tracking-wide">Loading...</span>
        </div>
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="fixed inset-0 z-50 bg-bg flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 text-center px-6">
          <span className="text-sm text-white/50">No libraries available</span>
          <button
            onClick={() => navigate('/')}
            className="px-4 py-2 text-xs text-white/70 bg-white/10 rounded-lg hover:bg-white/20 transition-colors"
          >
            Go back
          </button>
        </div>
      </div>
    );
  }

  const items: ViewerItem[] = images.map((img: GlobalImage) => ({
    key: img.content_hash,
    src: api.getImageUrl(img.library_name, img.content_hash),
    title: img.library_name,
  }));

  return (
    <Viewer
      items={items}
      onExit={() => navigate('/')}
      defaultTimer={180}
    />
  );
}

import { useNavigate, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import Viewer from '../components/Viewer';
import type { ViewerItem } from '../components/Viewer';
import type { BookPage } from '../types';

export default function BookPracticePage() {
  const navigate = useNavigate();
  const { slug } = useParams();

  const { data: allPages = [], isLoading } = useQuery({
    queryKey: ['books', 'practice', slug ?? 'all'],
    queryFn: () => api.getRandomBookPages(200, slug),
  });

  if (isLoading) {
    return (
      <div className="fixed inset-0 z-50 bg-bg flex items-center justify-center">
        <div className="w-5 h-5 border-2 border-white/20 border-t-white/70 rounded-full animate-spin" />
      </div>
    );
  }

  if (allPages.length === 0) {
    return (
      <div className="fixed inset-0 z-50 bg-bg flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <span className="text-sm text-white/50">No book pages available</span>
          <button onClick={() => navigate('/')} className="px-4 py-2 text-xs text-white/70 bg-white/10 rounded-lg hover:bg-white/20 transition-colors">Go back</button>
        </div>
      </div>
    );
  }

  const items: ViewerItem[] = allPages.map((p: BookPage) => ({
    key: `${p.slug}-${p.page_number}`,
    src: api.getBookPageUrl(p.slug, p.page_number),
    title: p.title,
    subtitle: String(p.page_number),
  }));

  return (
    <Viewer
      items={items}
      onExit={() => navigate('/')}
      defaultTimer={180}
    />
  );
}

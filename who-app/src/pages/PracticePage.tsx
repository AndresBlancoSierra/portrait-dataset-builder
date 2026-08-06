import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import Viewer from '../components/Viewer';
import type { ViewerItem } from '../components/Viewer';
import type { ImageWithMetadata } from '../types';

export default function PracticePage() {
  const { name } = useParams<{ name: string }>();
  const navigate = useNavigate();
  const decodedName = decodeURIComponent(name || '');

  const { data: images = [] } = useQuery({
    queryKey: ['images', decodedName],
    queryFn: () => api.getImages(decodedName),
    enabled: !!decodedName,
  });

  if (!decodedName || images.length === 0) return null;

  const items: ViewerItem[] = images.map((img: ImageWithMetadata) => ({
    key: img.content_hash,
    src: api.getImageUrl(decodedName, img.content_hash),
    title: decodedName,
    deleteFn() {
      api.deleteImage(decodedName, img.content_hash);
    },
  }));

  return (
    <Viewer
      items={items}
      onExit={() => navigate('/')}
    />
  );
}

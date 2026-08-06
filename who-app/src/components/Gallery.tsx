import { VirtuosoGrid } from 'react-virtuoso';
import { motion } from 'framer-motion';
import { useAppStore } from '../store';
import { api } from '../api/client';
import type { ImageWithMetadata } from '../types';
import { GallerySkeleton, EmptyState } from './Skeletons';
import { ImageOff, AlertCircle, Trash2 } from 'lucide-react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';

function GalleryItem({
  image,
  libraryName,
  allImages,
}: {
  image: ImageWithMetadata;
  libraryName: string;
  allImages: ImageWithMetadata[];
}) {
  const { openViewer } = useAppStore();
  const queryClient = useQueryClient();
  const [hovered, setHovered] = useState(false);

  const deleteMut = useMutation({
    mutationFn: () => api.deleteImage(libraryName, image.content_hash),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['images', libraryName] });
    },
  });

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.15 }}
      className="relative group"
      style={{ aspectRatio: '3/4' }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div
        className="w-full h-full rounded-lg overflow-hidden cursor-pointer bg-bg-card border border-border hover:border-border-hover transition-colors"
        onClick={() => openViewer(image, libraryName, allImages)}
        role="button"
        tabIndex={0}
        aria-label={`View image ${image.id}`}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            openViewer(image, libraryName, allImages);
          }
        }}
      >
        <img
          src={api.getImageUrl(libraryName, image.content_hash)}
          alt=""
          className="w-full h-full object-cover"
          loading="lazy"
        />
        {image.quality && (
          <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/70 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
            <span className="text-xs text-white/80 tabular-nums">
              {Math.round(image.quality.final_score * 100)}
            </span>
          </div>
        )}
      </div>
      <button
        onClick={(e) => {
          e.stopPropagation();
          deleteMut.mutate();
        }}
        style={{
          position: 'absolute',
          top: 6,
          right: 6,
          padding: 4,
          borderRadius: 4,
          opacity: hovered ? 1 : 0,
          background: 'rgba(0,0,0,0.5)',
          border: 'none',
          cursor: 'pointer',
          color: '#fff',
          transition: 'opacity 0.15s',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
        title="Delete image"
        aria-label={`Delete image ${image.id}`}
      >
        <Trash2 size={12} />
      </button>
    </motion.div>
  );
}

export function Gallery({
  images,
  isLoading,
  error,
  libraryName,
}: {
  images: ImageWithMetadata[];
  isLoading: boolean;
  error: Error | null;
  libraryName: string;
}) {
  if (isLoading) return <GallerySkeleton />;

  if (error) {
    return (
      <EmptyState
        icon={<AlertCircle size={40} style={{ color: '#dc2626' }} />}
        title="Failed to load images"
        description="Something went wrong loading this library. Please try again."
      />
    );
  }

  if (images.length === 0) {
    return (
      <EmptyState
        icon={<ImageOff size={40} />}
        title="No images found"
        description="Try adjusting your filters or build a new library."
      />
    );
  }

  return (
    <div className="h-full p-4">
      <VirtuosoGrid
        totalCount={images.length}
        overscan={200}
        itemContent={(index) => (
          <GalleryItem image={images[index]} libraryName={libraryName} allImages={images} />
        )}
        components={{
          Item: ({ children, ...props }) => (
            <div
              {...props}
              className="p-1.5"
              style={{ ...props.style, flex: 'none' }}
            >
              {children}
            </div>
          ),
          List: ({ children, ...props }) => (
            <div
              {...props}
              className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-0"
              style={{ ...props.style }}
            >
              {children}
            </div>
          ),
        }}
      />
    </div>
  );
}

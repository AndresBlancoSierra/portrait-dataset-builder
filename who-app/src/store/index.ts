import { create } from 'zustand';
import type { Library, ImageWithMetadata } from '../types';

interface AppState {
  activeLibrary: Library | null;
  setActiveLibrary: (lib: Library | null) => void;

  selectedImage: ImageWithMetadata | null;
  setSelectedImage: (img: ImageWithMetadata | null) => void;

  sidebarOpen: boolean;
  toggleSidebar: () => void;

  viewerOpen: boolean;
  viewerLibraryName: string;
  viewerImages: ImageWithMetadata[];
  viewerIndex: number;
  openViewer: (
    img: ImageWithMetadata,
    libraryName: string,
    images?: ImageWithMetadata[],
  ) => void;
  navigateViewer: (direction: -1 | 1) => void;
  removeViewerImage: (contentHash: string) => void;
  closeViewer: () => void;

  practiceMode: boolean;
  setPracticeMode: (v: boolean) => void;

  uiHidden: boolean;
  toggleUi: () => void;

  activeBuilds: Set<string>;
  addActiveBuild: (name: string) => void;
  removeActiveBuild: (name: string) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  activeLibrary: null,
  setActiveLibrary: (lib) => set({ activeLibrary: lib }),

  selectedImage: null,
  setSelectedImage: (img) => set({ selectedImage: img }),

  sidebarOpen: true,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  viewerOpen: false,
  viewerLibraryName: '',
  viewerImages: [],
  viewerIndex: 0,
  openViewer: (img, libraryName, images) => {
    const list = images ?? [img];
    const idx = list.findIndex((i) => i.content_hash === img.content_hash);
    set({
      viewerOpen: true,
      selectedImage: img,
      viewerLibraryName: libraryName,
      viewerImages: list,
      viewerIndex: idx >= 0 ? idx : 0,
    });
  },
  navigateViewer: (direction) => {
    const { viewerImages, viewerIndex } = get();
    if (viewerImages.length === 0) return;
    const next = viewerIndex + direction;
    if (next < 0 || next >= viewerImages.length) return;
    set({
      viewerIndex: next,
      selectedImage: viewerImages[next],
    });
  },
  removeViewerImage: (contentHash) => {
    const { viewerImages, viewerIndex } = get();
    const remaining = viewerImages.filter((i) => i.content_hash !== contentHash);
    if (remaining.length === 0) {
      set({ viewerOpen: false, viewerImages: [], selectedImage: null, viewerIndex: 0 });
      return;
    }
    const nextIdx = Math.min(viewerIndex, remaining.length - 1);
    set({
      viewerImages: remaining,
      viewerIndex: nextIdx,
      selectedImage: remaining[nextIdx],
    });
  },
  closeViewer: () => set({ viewerOpen: false }),

  practiceMode: false,
  setPracticeMode: (v) => set({ practiceMode: v, uiHidden: v }),

  uiHidden: false,
  toggleUi: () => set((s) => ({ uiHidden: !s.uiHidden })),

  activeBuilds: new Set<string>(),
  addActiveBuild: (name) =>
    set((s) => {
      const next = new Set(s.activeBuilds);
      next.add(name);
      return { activeBuilds: next };
    }),
  removeActiveBuild: (name) =>
    set((s) => {
      const next = new Set(s.activeBuilds);
      next.delete(name);
      return { activeBuilds: next };
    }),
}));

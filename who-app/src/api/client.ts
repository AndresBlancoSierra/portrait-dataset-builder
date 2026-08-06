import type { Library, ImageWithMetadata, PipelineProgress, CoverageData, Stats, BuildQueueStatus, BatchEnqueueResult, GlobalImage, Book, BookPage } from '../types';

const BASE = '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  if (res.status === 204 || res.headers.get('content-length') === '0') {
    return undefined as T;
  }
  return res.json();
}

export const api = {
  listLibraries: () => request<Library[]>('/libraries'),

  getLibrary: (name: string) =>
    request<Library>(`/libraries/${encodeURIComponent(name)}`),

  createLibrary: (name: string) =>
    request<{ name: string; status: string; build_job_id: number }>('/libraries', {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),

  deleteLibrary: (name: string) =>
    request<void>(`/libraries/${encodeURIComponent(name)}`, { method: 'DELETE' }),

  deleteImage: (name: string, hash: string) =>
    request<void>(`/libraries/${encodeURIComponent(name)}/images/${hash}`, { method: 'DELETE' }),

  getImages: (name: string, params?: Record<string, string>) => {
    const qs = params ? '?' + new URLSearchParams(params).toString() : '';
    return request<ImageWithMetadata[]>(`/libraries/${encodeURIComponent(name)}/images${qs}`);
  },

  getImage: (name: string, hash: string) =>
    request<ImageWithMetadata>(`/libraries/${encodeURIComponent(name)}/images/${hash}`),

  getImageUrl: (name: string, hash: string) =>
    `${BASE}/libraries/${encodeURIComponent(name)}/images/${hash}/file`,

  startBuild: (name: string) =>
    request<{ name: string; status: string; build_job_id: number }>(
      `/libraries/${encodeURIComponent(name)}/build`,
      { method: 'POST' },
    ),

  getBuildProgress: (name: string) =>
    request<PipelineProgress>(`/libraries/${encodeURIComponent(name)}/build/progress`),

  cancelBuild: (name: string) =>
    request<void>(`/libraries/${encodeURIComponent(name)}/cancel`, { method: 'POST' }),

  getCoverage: (name: string) =>
    request<CoverageData>(`/libraries/${encodeURIComponent(name)}/coverage`),

  getStats: (name: string) =>
    request<Stats>(`/libraries/${encodeURIComponent(name)}/stats`),

  getReviewQueue: (name: string) =>
    request<ImageWithMetadata[]>(`/libraries/${encodeURIComponent(name)}/review`),

  reviewImage: (name: string, hash: string, accepted: boolean) =>
    request<void>(`/libraries/${encodeURIComponent(name)}/review/${hash}`, {
      method: 'POST',
      body: JSON.stringify({ accepted }),
    }),

  getBuildQueue: () => request<BuildQueueStatus>('/builds/queue'),

  batchEnqueue: (names: string[]) =>
    request<BatchEnqueueResult>('/queue/batch', {
      method: 'POST',
      body: JSON.stringify({ names }),
    }),

  cancelQueueJob: (jobId: number) =>
    request<{ ok: boolean }>(`/queue/${jobId}/cancel`, { method: 'POST' }),

  retryQueueJob: (jobId: number) =>
    request<{ ok: boolean }>(`/queue/${jobId}/retry`, { method: 'POST' }),

  removeQueueJob: (jobId: number) =>
    request<{ ok: boolean }>(`/queue/${jobId}`, { method: 'DELETE' }),

  pauseQueue: () =>
    request<{ ok: boolean; paused: boolean }>('/queue/pause', { method: 'POST' }),

  resumeQueue: () =>
    request<{ ok: boolean; paused: boolean }>('/queue/resume', { method: 'POST' }),

  getRandomGlobal: (count: number = 100) =>
    request<GlobalImage[]>(`/practice/random-global?count=${count}`),

  listBooks: () =>
    request<Book[]>('/books'),

  getRandomBookPages: (count: number = 100, book?: string) =>
    request<BookPage[]>(`/books/pages/random?count=${count}${book ? '&book=' + encodeURIComponent(book) : ''}`),

  getBookPageUrl: (slug: string, pageNum: number) =>
    `${BASE}/books/pages/${encodeURIComponent(slug)}/${pageNum}/file`,
};

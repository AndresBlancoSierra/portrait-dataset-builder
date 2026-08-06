import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import HomePage from '../HomePage';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { Library } from '../../types';

function makeLib(overrides: Partial<Library> = {}): Library {
  return {
    name: 'Test Person',
    image_count: 100,
    quality_score: 0.85,
    coverage_score: 0.60,
    updated_at: '',
    thumbnail_hash: null,
    status: 'ready',
    build: { id: 1, status: 'completed', current_stage: null, stage_label: null, items_processed: 0, items_total: 0, error: null, started_at: null, completed_at: null, created_at: null },
    ...overrides,
  };
}

function LocationDisplay() {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

function renderHome(libraries: Library[] = []) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });

  vi.spyOn(globalThis, 'fetch').mockImplementation((input: RequestInfo | URL) => {
    const u = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    if (u.includes('/api/libraries')) {
      return Promise.resolve(new Response(JSON.stringify(libraries)));
    }
    if (u.includes('/api/builds/queue')) {
      return Promise.resolve(new Response(JSON.stringify({ jobs: [], active_job: null, queue_paused: false, max_concurrent: 1 })));
    }
    return Promise.resolve(new Response(JSON.stringify([])));
  });

  return render(
    <MemoryRouter initialEntries={['/']}>
      <QueryClientProvider client={queryClient}>
        <LocationDisplay />
        <HomePage />
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

describe('HomePage', () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('shows shuffle button when libraries exist', async () => {
    renderHome([makeLib()]);
    await waitFor(() => {
      expect(screen.getByLabelText('Random practice')).toBeInTheDocument();
    });
  });

  it('does not show shuffle button when no libraries', async () => {
    renderHome([]);
    await waitFor(() => {
      expect(screen.queryByText('Libraries')).toBeInTheDocument();
    });
    expect(screen.queryByLabelText('Random practice')).not.toBeInTheDocument();
  });

  it('shuffle button navigates to /practice/random', async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderHome([makeLib()]);
    await waitFor(() => {
      expect(screen.getByLabelText('Random practice')).toBeInTheDocument();
    });
    await user.click(screen.getByLabelText('Random practice'));
    await waitFor(() => {
      expect(screen.getByTestId('location')).toHaveTextContent('/practice/random');
    });
  });

  it('does not show Settings button', async () => {
    renderHome([]);
    await waitFor(() => {
      expect(screen.queryByText('Libraries')).toBeInTheDocument();
    });
    expect(screen.queryByLabelText('Settings')).not.toBeInTheDocument();
  });
});

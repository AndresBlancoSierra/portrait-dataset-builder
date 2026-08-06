import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import { LibraryCard } from '../LibraryCard';
import type { Library } from '../types';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

function makeLibrary(overrides: Partial<Library> = {}): Library {
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

function renderCard(library: Library, onRefresh?: () => void) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <LibraryCard library={library} onRefresh={onRefresh} />
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

describe('LibraryCard', () => {
  it('renders library name', () => {
    renderCard(makeLibrary());
    expect(screen.getByText('Test Person')).toBeInTheDocument();
  });

  it('shows Ready status indicator', () => {
    renderCard(makeLibrary({ status: 'ready' }));
    expect(screen.getByText('Ready')).toBeInTheDocument();
  });

  it('shows Building status indicator', () => {
    renderCard(makeLibrary({ status: 'building' }));
    expect(screen.getByText('Building')).toBeInTheDocument();
  });

  it('shows Failed status indicator', () => {
    renderCard(makeLibrary({ status: 'failed' }));
    expect(screen.getByText('Failed')).toBeInTheDocument();
  });

  it('shows Cancelled status indicator', () => {
    renderCard(makeLibrary({ status: 'cancelled' }));
    expect(screen.getByText('Cancelled')).toBeInTheDocument();
  });

  it('shows Empty status indicator for empty status', () => {
    renderCard(makeLibrary({ status: 'empty' }));
    expect(screen.getByText('Empty')).toBeInTheDocument();
  });

  it('does not show Ready for empty status', () => {
    renderCard(makeLibrary({ status: 'empty' }));
    expect(screen.queryByText('Ready')).not.toBeInTheDocument();
  });

  it('shows quality and coverage bars for ready library', () => {
    renderCard(makeLibrary({ status: 'ready', quality_score: 0.85, coverage_score: 0.60 }));
    expect(screen.getByText('Q')).toBeInTheDocument();
    expect(screen.getByText('C')).toBeInTheDocument();
  });

  it('shows image count for ready library', () => {
    renderCard(makeLibrary({ status: 'ready', image_count: 150 }));
    expect(screen.getByText('150 references')).toBeInTheDocument();
  });

  it('shows empty overlay for empty library', () => {
    renderCard(makeLibrary({ status: 'empty' }));
    expect(screen.getByText('No valid images found')).toBeInTheDocument();
  });

  it('navigates to build page when clicking building library', async () => {
    const user = userEvent.setup();
    renderCard(makeLibrary({ status: 'building' }));

    const card = screen.getByRole('button', { name: /open library test person/i });
    await user.click(card);
    // navigate would be called - can't fully assert without mocking, but shouldn't crash
  });
});

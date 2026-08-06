import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import { FiltersPanel } from '../FiltersPanel';

describe('FiltersPanel', () => {
  const onChange = vi.fn();

  beforeEach(() => {
    onChange.mockClear();
  });

  it('renders all filter sections', () => {
    render(<FiltersPanel filters={{}} onChange={onChange} />);
    expect(screen.getByText('Horizontal Angle')).toBeInTheDocument();
    expect(screen.getByText('Vertical Pose')).toBeInTheDocument();
    expect(screen.getByText('Expression')).toBeInTheDocument();
    expect(screen.getByText('Lighting')).toBeInTheDocument();
    expect(screen.getByText('Quality')).toBeInTheDocument();
  });

  it('shows correct horizontal angle options', () => {
    render(<FiltersPanel filters={{}} onChange={onChange} />);
    expect(screen.getByText('Frontal')).toBeInTheDocument();
    expect(screen.getByText('\u00be Left')).toBeInTheDocument();
    expect(screen.getByText('\u00be Right')).toBeInTheDocument();
    expect(screen.getByText('Profile Left')).toBeInTheDocument();
    expect(screen.getByText('Profile Right')).toBeInTheDocument();
  });

  it('shows correct vertical pose options', () => {
    render(<FiltersPanel filters={{}} onChange={onChange} />);
    // "Neutral" exists in both Vertical Pose and Expression, so use getAllByText
    const neutralElements = screen.getAllByText('Neutral');
    expect(neutralElements.length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText('Looking Up')).toBeInTheDocument();
    expect(screen.getByText('Looking Down')).toBeInTheDocument();
  });

  it('shows correct expression options', () => {
    render(<FiltersPanel filters={{}} onChange={onChange} />);
    const neutralElements = screen.getAllByText('Neutral');
    expect(neutralElements.length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText('Smile')).toBeInTheDocument();
    expect(screen.getByText('Laugh')).toBeInTheDocument();
    expect(screen.getByText('Speaking')).toBeInTheDocument();
    expect(screen.queryByText('Serious')).not.toBeInTheDocument();
  });

  it('shows correct lighting options', () => {
    render(<FiltersPanel filters={{}} onChange={onChange} />);
    expect(screen.getByText('Dark')).toBeInTheDocument();
    expect(screen.getByText('Bright')).toBeInTheDocument();
    expect(screen.getByText('Balanced')).toBeInTheDocument();
    expect(screen.queryByText('Front')).not.toBeInTheDocument();
    expect(screen.queryByText('Side')).not.toBeInTheDocument();
  });

  it('shows correct quality options', () => {
    render(<FiltersPanel filters={{}} onChange={onChange} />);
    expect(screen.getByText('High (80+)')).toBeInTheDocument();
    expect(screen.getByText('Medium (50-80)')).toBeInTheDocument();
    expect(screen.getByText('Low (<50)')).toBeInTheDocument();
  });

  it('toggles expression filter on click', async () => {
    const user = userEvent.setup();
    render(<FiltersPanel filters={{}} onChange={onChange} />);

    await user.click(screen.getByText('Smile'));
    expect(onChange).toHaveBeenCalledWith({ expression: 'smile' });
  });

  it('adds multiple values to same filter', async () => {
    const user = userEvent.setup();
    render(<FiltersPanel filters={{ expression: 'smile' }} onChange={onChange} />);

    await user.click(screen.getByText('Laugh'));
    expect(onChange).toHaveBeenCalledWith({ expression: 'smile,laugh' });
  });

  it('removes value from multi-select filter', async () => {
    const user = userEvent.setup();
    render(<FiltersPanel filters={{ expression: 'smile,laugh' }} onChange={onChange} />);

    await user.click(screen.getByText('Smile'));
    expect(onChange).toHaveBeenCalledWith({ expression: 'laugh' });
  });

  it('removes filter key when all values deselected', async () => {
    const user = userEvent.setup();
    render(<FiltersPanel filters={{ expression: 'smile' }} onChange={onChange} />);

    await user.click(screen.getByText('Smile'));
    expect(onChange).toHaveBeenCalledWith({});
  });

  it('shows Clear all button when filters active', () => {
    render(<FiltersPanel filters={{ expression: 'smile' }} onChange={onChange} />);
    expect(screen.getByText('Clear all filters')).toBeInTheDocument();
  });

  it('hides Clear all button when no filters active', () => {
    render(<FiltersPanel filters={{}} onChange={onChange} />);
    expect(screen.queryByText('Clear all filters')).not.toBeInTheDocument();
  });

  it('clears all filters on Clear all click', async () => {
    const user = userEvent.setup();
    render(<FiltersPanel filters={{ expression: 'smile', lighting: 'dark' }} onChange={onChange} />);

    await user.click(screen.getByText('Clear all filters'));
    expect(onChange).toHaveBeenCalledWith({});
  });

  it('marks selected checkboxes correctly', () => {
    render(<FiltersPanel filters={{ expression: 'smile' }} onChange={onChange} />);
    const smileCheckbox = screen.getAllByRole('checkbox').find(
      (el) => el.getAttribute('aria-label') === 'Smile' || el.textContent?.includes('Smile'),
    );
    expect(smileCheckbox).toHaveAttribute('aria-checked', 'true');
  });

  it('toggles vertical pose filter on click', async () => {
    const user = userEvent.setup();
    render(<FiltersPanel filters={{}} onChange={onChange} />);

    await user.click(screen.getByText('Looking Up'));
    expect(onChange).toHaveBeenCalledWith({ vertical_pose: 'looking_up' });
  });

  it('toggles horizontal pose filter on click', async () => {
    const user = userEvent.setup();
    render(<FiltersPanel filters={{}} onChange={onChange} />);

    await user.click(screen.getByText('Profile Left'));
    expect(onChange).toHaveBeenCalledWith({ horizontal_pose: 'profile_left' });
  });

  it('supports multi-filter combinations via controlled prop', async () => {
    const user = userEvent.setup();
    // When filters prop already contains horizontal_pose, toggling expression adds it
    render(<FiltersPanel filters={{ horizontal_pose: 'three_quarter_left' }} onChange={onChange} />);
    // Verify horizontal pose is shown as selected
    const threeQuarterLeftBtn = screen.getAllByRole('checkbox').find(
      (el) => el.getAttribute('aria-label')?.includes('\u00be Left') || el.textContent?.includes('\u00be Left'),
    );
    expect(threeQuarterLeftBtn).toHaveAttribute('aria-checked', 'true');
    // Now click expression - it should include horizontal_pose in the merged result
    await user.click(screen.getByText('Smile'));
    expect(onChange).toHaveBeenCalledWith({ horizontal_pose: 'three_quarter_left', expression: 'smile' });
  });
});

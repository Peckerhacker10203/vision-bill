import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, expect, it, vi } from 'vitest';
import ReviewFindings from './ReviewFindings.svelte';
import type { ReviewFinding } from '$lib/types';
const findings: ReviewFinding[] = [
	{
		id: 'i',
		code: 'original_unavailable',
		severity: 'integrity' as const,
		field_path: 'originals',
		source_refs: [],
		comparison: {}
	},
	{
		id: 'r',
		code: 'total_mismatch',
		severity: 'review' as const,
		field_path: 'total',
		source_refs: [],
		comparison: { expected: '1.00000001' }
	}
];
afterEach(cleanup);
it('only review findings accept comments and reset on a new review version', async () => {
	const onChange = vi.fn();
	const view = render(ReviewFindings, { findings, revision: 1, onChange });
	expect(screen.getAllByRole('textbox')).toHaveLength(1);
	await fireEvent.input(screen.getByLabelText('Review comment: total_mismatch'), {
		target: { value: 'Printed total checked' }
	});
	expect(onChange).toHaveBeenLastCalledWith([
		{ finding_id: 'r', comment: 'Printed total checked' }
	]);
	await view.rerender({ findings, revision: 2, onChange });
	expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('');
	expect(onChange).toHaveBeenLastCalledWith([]);
});
it('clears prior comments as soon as material editing starts', async () => {
	const onChange = vi.fn();
	const view = render(ReviewFindings, { findings, revision: 1, onChange });
	await fireEvent.input(screen.getByRole('textbox'), { target: { value: 'Old acknowledgement' } });
	await view.rerender({ findings, revision: 1, onChange, editable: false });
	await view.rerender({ findings, revision: 1, onChange, editable: true });
	expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('');
	expect(onChange).toHaveBeenLastCalledWith([]);
});

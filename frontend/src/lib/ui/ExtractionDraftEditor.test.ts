import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import ExtractionDraftEditor from './ExtractionDraftEditor.svelte';
import { api, ApiError } from '$lib/api/client';
import { queryClient } from '$lib/query/client';
const draft = {
	image_id: 'job',
	revision: 7,
	field_sources: {},
	overlap_decisions: [],
	document: {
		merchant_name: null,
		subtotal: null,
		total: null,
		currency: null,
		line_items: [
			{ description: null, quantity: null, unit_price: null, total_price: null, tags: null }
		]
	},
	amount_policy: {
		tax_basis: 'unknown' as const,
		discount_treatment: 'unknown' as const,
		tip_treatment: 'unknown' as const,
		subtotal_origin: 'unknown' as const,
		tax_information: 'missing' as const
	},
	findings: [],
	source_views: [],
	finalized_receipt_id: null
};
beforeEach(() => {
	queryClient.clear();
	vi.spyOn(api, 'listTags').mockResolvedValue([]);
	vi.stubGlobal('matchMedia', () => ({ matches: false }));
	vi.spyOn(api, 'getImage').mockResolvedValue({ additional_images: [] } as never);
	vi.spyOn(api, 'getDraft').mockResolvedValue(draft);
});
afterEach(() => {
	queryClient.clear();
	cleanup();
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
});
it('partial saves preserve nulls, missing fields and actual revision', async () => {
	const patch = vi.spyOn(api, 'patchDraft').mockResolvedValue({
		...draft,
		revision: 8,
		document: { ...draft.document, merchant_name: 'Shop' }
	});
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Shop' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	expect(patch).toHaveBeenCalledWith('job', {
		expected_revision: 7,
		changes: { merchant_name: 'Shop' }
	});
	expect((screen.getByLabelText('Currency') as HTMLInputElement).value).toBe('');
	expect((screen.getByLabelText('Qty') as HTMLInputElement).value).toBe('');
});
it('conflicts preserve dirty input and original expected revision', async () => {
	const patch = vi.spyOn(api, 'patchDraft').mockRejectedValue(new ApiError(409, 'Draft conflict'));
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'My edit' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	await screen.findByRole('alert');
	expect((screen.getByLabelText('Merchant') as HTMLInputElement).value).toBe('My edit');
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	expect(patch.mock.calls[1][1].expected_revision).toBe(7);
});
it('shows required missing fields without attempting incomplete finalization', async () => {
	const finalize = vi.spyOn(api, 'finalizeDraft');
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Merchant');
	await fireEvent.click(screen.getByRole('button', { name: 'Finalize draft' }));
	expect(screen.getByRole('alert').textContent).toContain('merchant_name');
	expect(finalize).not.toHaveBeenCalled();
});

it('finalizes the current saved revision with one stable key across ambiguous responses', async () => {
	const complete = {
		...draft,
		document: {
			confidence: 90,
			merchant_name: 'Shop',
			date: '2026-01-01',
			currency: 'EUR',
			category: 'grocery' as const,
			line_items: [],
			taxes: [],
			subtotal: '0',
			discount_total: '0',
			tax_total: '0',
			total: '0',
			payment_method: 'cash' as const
		}
	};
	vi.mocked(api.getDraft).mockResolvedValue(complete);
	const finalize = vi
		.spyOn(api, 'finalizeDraft')
		.mockRejectedValueOnce(new ApiError(0, 'Network error'))
		.mockResolvedValue({
			...complete.document,
			id: 'receipt',
			status: 'unverified',
			verified: false
		} as never);
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Merchant');
	await fireEvent.click(screen.getByRole('button', { name: 'Finalize draft' }));
	await screen.findByRole('alert');
	await fireEvent.click(screen.getByRole('button', { name: 'Finalize draft' }));
	expect(finalize).toHaveBeenCalledTimes(2);
	expect(finalize.mock.calls[0][1]).toEqual(finalize.mock.calls[1][1]);
	expect(finalize.mock.calls[1][1]).toMatchObject({
		expected_revision: 7,
		request_key: expect.any(String)
	});
});
it('ignores a late save after switching to another job', async () => {
	let done!: (value: typeof draft) => void;
	vi.spyOn(api, 'patchDraft').mockImplementation(
		() =>
			new Promise((resolve) => {
				done = resolve as typeof done;
			})
	);
	const view = render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'old edit' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	vi.mocked(api.getDraft).mockResolvedValue({
		...draft,
		image_id: 'other',
		document: { merchant_name: 'Other job' }
	});
	await view.rerender({ id: 'other' });
	await screen.findByDisplayValue('Other job');
	done({ ...draft, revision: 8 });
	expect((screen.getByLabelText('Merchant') as HTMLInputElement).value).toBe('Other job');
});

it('uses the owner tag vocabulary when reusing tag controls in a nullable draft', async () => {
	const tags = vi.mocked(api.listTags).mockResolvedValue(['food']);
	vi.mocked(api.getDraft).mockResolvedValue({
		...draft,
		document: { ...draft.document, line_items: [{ description: 'Item', tags: ['food'] }] }
	});
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Merchant');
	expect(tags).toHaveBeenCalledOnce();
	expect(screen.queryByText('Suggested')).toBeNull();
});

import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/svelte';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { api, ApiError } from '$lib/api/client';
import { queryClient } from '$lib/query/client';
import type { ReceiptRow, ReceiptWithDetails } from '$lib/types';
import Page from './+page.svelte';
import { page } from '$app/state';
import { tick } from 'svelte';
vi.mock('$app/state', async () => {
	const { SvelteURL } = await import('svelte/reactivity');
	return {
		page: { url: new SvelteURL('http://localhost/receipts/00000000-0000-4000-8000-000000000001') }
	};
});
vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
const row: ReceiptRow = {
	id: '00000000-0000-4000-8000-000000000001',
	confidence: 90,
	merchant_name: 'Shop',
	merchant_address: null,
	receipt_number: null,
	date: '2026-01-01',
	time: null,
	currency: 'EUR',
	category: 'grocery',
	subtotal: '2',
	discount_total: '0',
	tax_total: '0',
	tip: null,
	total: '2',
	payment_method: 'cash',
	created_at: null,
	status: 'unverified',
	image_id: null,
	thumbnail_path: null,
	verified: false,
	user_id: null,
	operation_revision: 7,
	review_revision: 3,
	amount_policy: {
		tax_basis: 'gross',
		discount_treatment: 'included_in_lines',
		tip_treatment: 'none',
		subtotal_origin: 'printed',
		tax_information: 'explicit_zero'
	},
	findings: [],
	field_sources: {},
	overlap_decisions: [],
	acknowledgements: [],
	checks_version: '1',
	verification_source: null
};
const detail = (receipt = row): ReceiptWithDetails => ({
	receipt,
	line_items: [],
	taxes: [],
	collection_ids: [],
	image_path: null,
	classification_enrichment: { state: 'not_recorded', reason_code: null, missing_fields: [] }
});
beforeEach(() => {
	queryClient.clear();
	page.url.pathname = '/receipts/00000000-0000-4000-8000-000000000001';
	vi.spyOn(api, 'getReceipt').mockResolvedValue(detail());
	vi.spyOn(api, 'listTags').mockResolvedValue([]);
	vi.spyOn(api, 'listCollections').mockResolvedValue([]);
	vi.spyOn(api, 'listSpendingCategories').mockResolvedValue([]);
});
afterEach(() => {
	cleanup();
	queryClient.clear();
	vi.restoreAllMocks();
});
it('keeps dirty form and expected operation revision on a PUT conflict', async () => {
	const update = vi.spyOn(api, 'updateReceipt').mockRejectedValue(new ApiError(409, 'stale'));
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'My edit' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await waitFor(() => expect(update).toHaveBeenCalledOnce());
	expect((screen.getByLabelText('Merchant') as HTMLInputElement).value).toBe('My edit');
	await waitFor(() =>
		expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(false)
	);
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(update.mock.calls[1][1].expected_operation_revision).toBe(7);
});
it('does not verify new material findings using pre-save comments', async () => {
	const findings = [
		{
			id: 'fresh',
			code: 'total_mismatch',
			severity: 'review' as const,
			field_path: 'total',
			source_refs: [],
			comparison: {}
		}
	];
	const saved = {
		...row,
		merchant_name: 'Changed',
		operation_revision: 8,
		review_revision: 4,
		findings
	};
	vi.spyOn(api, 'updateReceipt').mockResolvedValue(saved);
	vi.spyOn(api, 'getReceipt').mockResolvedValueOnce(detail()).mockResolvedValue(detail(saved));
	const verify = vi.spyOn(api, 'verifyReceipt');
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Changed' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save & verify' }));
	await screen.findByLabelText(
		'Review comment: Additional diagnostic: total mismatch. Check the original or processing status.'
	);
	expect(verify).not.toHaveBeenCalled();
	expect(screen.queryByRole('button', { name: 'Save' })).toBeNull();
});
it('uses successful PUT versions for verification and keeps save successful if verify fails', async () => {
	const saved = { ...row, merchant_name: 'Saved', operation_revision: 8, review_revision: 4 };
	vi.spyOn(api, 'updateReceipt').mockResolvedValue(saved);
	vi.spyOn(api, 'getReceipt').mockResolvedValueOnce(detail()).mockResolvedValue(detail(saved));
	const verify = vi
		.spyOn(api, 'verifyReceipt')
		.mockRejectedValue(new ApiError(422, 'missing original'));
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Saved' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save & verify' }));
	await waitFor(() =>
		expect(verify).toHaveBeenCalledWith(row.id, {
			expected_operation_revision: 8,
			expected_review_revision: 4,
			acknowledgements: []
		})
	);
	await waitFor(() => expect(screen.queryByRole('button', { name: 'Save' })).toBeNull());
	expect((screen.getByLabelText('Merchant') as HTMLInputElement).value).toBe('Saved');
});

it('has no acknowledgement control for missing originals and disables verification', async () => {
	vi.mocked(api.getReceipt).mockResolvedValue(
		detail({
			...row,
			findings: [
				{
					id: 'missing',
					code: 'missing_original',
					severity: 'integrity',
					field_path: 'originals',
					source_refs: [],
					comparison: {}
				}
			]
		})
	);
	render(Page);
	await screen.findByText(/Verification blocked: The original photo is missing/);
	expect(screen.queryByRole('textbox', { name: /Review comment/ })).toBeNull();
	expect((screen.getByRole('button', { name: 'Verify' }) as HTMLButtonElement).disabled).toBe(true);
});
it('displays server revocation after editing a verified receipt', async () => {
	vi.mocked(api.getReceipt)
		.mockResolvedValueOnce(detail({ ...row, status: 'verified', verified: true }))
		.mockResolvedValue(
			detail({ ...row, merchant_name: 'Edited', operation_revision: 8, review_revision: 4 })
		);
	vi.spyOn(api, 'updateReceipt').mockResolvedValue({
		...row,
		merchant_name: 'Edited',
		operation_revision: 8,
		review_revision: 4
	});
	render(Page);
	await screen.findByText('Verified');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Edited' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await screen.findByRole('button', { name: 'Verify' });
	expect(screen.queryByText('Verified')).toBeNull();
});

function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (error: unknown) => void;
	const promise = new Promise<T>((ok, no) => {
		resolve = ok;
		reject = no;
	});
	return { promise, resolve, reject };
}
const second = { ...row, id: '00000000-0000-4000-8000-000000000002', merchant_name: 'Second' };
async function navigateSecond() {
	page.url.pathname = `/receipts/${second.id}`;
	await tick();
	await screen.findByDisplayValue('Second');
}
it('resets busy state and ignores a late PUT across receipt navigation', async () => {
	const pending = deferred<ReceiptRow>();
	vi.spyOn(api, 'updateReceipt').mockReturnValue(pending.promise);
	vi.mocked(api.getReceipt).mockImplementation(async (id) =>
		detail(id === second.id ? second : row)
	);
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'A edit' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await navigateSecond();
	expect((screen.getByRole('button', { name: 'Verify' }) as HTMLButtonElement).disabled).toBe(
		false
	);
	pending.reject(new ApiError(409, 'A conflict'));
	await tick();
	expect(screen.queryByRole('alert')).toBeNull();
});
it('resets a pending verify and its dialog across receipt navigation', async () => {
	const pending = deferred<ReceiptRow>();
	vi.spyOn(api, 'verifyReceipt').mockReturnValue(pending.promise);
	vi.mocked(api.getReceipt).mockImplementation(async (id) =>
		detail(id === second.id ? second : row)
	);
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.click(screen.getByRole('button', { name: 'Verify' }));
	await fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Verify' }));
	await waitFor(() => expect(api.verifyReceipt).toHaveBeenCalledOnce());
	await navigateSecond();
	expect(screen.queryByRole('dialog')).toBeNull();
	expect((screen.getByRole('button', { name: 'Verify' }) as HTMLButtonElement).disabled).toBe(
		false
	);
	pending.reject(new ApiError(409, 'A conflict'));
	await tick();
	expect(screen.queryByRole('alert')).toBeNull();
});
it('fences discard-reload errors to their originating receipt view', async () => {
	vi.spyOn(api, 'updateReceipt').mockRejectedValue(new ApiError(409, 'stale'));
	const pending = deferred<ReceiptWithDetails>();
	let reads = 0;
	vi.mocked(api.getReceipt).mockImplementation((id) =>
		id === second.id
			? Promise.resolve(detail(second))
			: ++reads === 1
				? Promise.resolve(detail())
				: pending.promise
	);
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'A edit' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await screen.findByRole('alert');
	await fireEvent.click(
		screen.getByRole('button', { name: 'Discard edits and load current version' })
	);
	await navigateSecond();
	pending.reject(new ApiError(0, 'A reload failed'));
	await tick();
	expect(screen.queryByRole('alert')).toBeNull();
	expect((screen.getByRole('button', { name: 'Verify' }) as HTMLButtonElement).disabled).toBe(
		false
	);
});
const item = {
	id: 'line',
	receipt_id: row.id,
	description: 'Item',
	original_description: null,
	quantity: 1,
	unit_price: '2',
	total_price: '2',
	spending_category: 'unknown',
	category_source: 'unknown' as const,
	tags: [] as string[]
};
it('waits for deferred tag creation before saving, then reloads the submitted tag', async () => {
	vi.mocked(api.listTags).mockResolvedValue(['food']);
	const pending = deferred<Awaited<ReturnType<typeof api.createTag>>>();
	vi.spyOn(api, 'createTag').mockReturnValue(pending.promise);
	const update = vi
		.spyOn(api, 'updateReceipt')
		.mockResolvedValue({ ...row, merchant_name: 'Changed', operation_revision: 8 });
	vi.mocked(api.getReceipt)
		.mockResolvedValueOnce({ ...detail(), line_items: [item] })
		.mockResolvedValue({
			...detail({ ...row, merchant_name: 'Changed', operation_revision: 8 }),
			line_items: [{ ...item, tags: ['new tag'] }]
		});
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Changed' } });
	await fireEvent.click(screen.getByText('Optional tags'));
	await fireEvent.click(screen.getByRole('button', { name: 'Tags' }));
	await fireEvent.input(screen.getByPlaceholderText('Search tags…'), {
		target: { value: 'new tag' }
	});
	await fireEvent.click(screen.getByRole('option', { name: 'Create “new tag”' }));
	expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(true);
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(update).not.toHaveBeenCalled();
	pending.resolve({ name: 'new tag', created: true });
	await waitFor(() =>
		expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(false)
	);
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await waitFor(() => expect(update).toHaveBeenCalledOnce());
	expect(update.mock.calls[0][1].line_items[0].tags).toEqual(['new tag']);
	await waitFor(() => expect(screen.queryByRole('button', { name: 'Save' })).toBeNull());
	await fireEvent.click(screen.getByText('Optional tags'));
	expect(screen.getByText('new tag')).toBeTruthy();
});
it('waits for deferred collection creation before saving its membership', async () => {
	const pending = deferred<import('$lib/types').Collection>();
	vi.spyOn(api, 'createCollection').mockReturnValue(pending.promise);
	const update = vi
		.spyOn(api, 'updateReceipt')
		.mockResolvedValue({ ...row, merchant_name: 'Changed', operation_revision: 8 });
	vi.mocked(api.getReceipt)
		.mockResolvedValueOnce(detail())
		.mockResolvedValue({
			...detail({ ...row, merchant_name: 'Changed', operation_revision: 8 }),
			collection_ids: ['created']
		});
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Changed' } });
	const picker = screen.getByPlaceholderText('Search collections…');
	await fireEvent.click(picker);
	await fireEvent.input(picker, { target: { value: 'New' } });
	await fireEvent.keyDown(picker, { key: 'Enter' });
	expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(true);
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(update).not.toHaveBeenCalled();
	pending.resolve({
		id: 'created',
		name: 'New',
		color: null,
		start_date: null,
		end_date: null,
		active: true,
		created_at: null
	});
	await waitFor(() =>
		expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(false)
	);
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await waitFor(() => expect(update).toHaveBeenCalledOnce());
	expect(update.mock.calls[0][1].collection_ids).toEqual(['created']);
});
it('does not attach a late created collection to the next receipt', async () => {
	const pending = deferred<import('$lib/types').Collection>();
	vi.spyOn(api, 'createCollection').mockReturnValue(pending.promise);
	const update = vi
		.spyOn(api, 'updateReceipt')
		.mockResolvedValue({ ...second, merchant_name: 'B edit', operation_revision: 8 });
	vi.mocked(api.getReceipt).mockImplementation(async (id) =>
		detail(id === second.id ? second : row)
	);
	render(Page);
	await screen.findByLabelText('Merchant');
	const picker = screen.getByPlaceholderText('Search collections…');
	await fireEvent.click(picker);
	await fireEvent.input(picker, { target: { value: 'For A' } });
	await fireEvent.keyDown(picker, { key: 'Enter' });
	await navigateSecond();
	pending.resolve({
		id: 'for-a',
		name: 'For A',
		color: null,
		start_date: null,
		end_date: null,
		active: true,
		created_at: null
	});
	await tick();
	expect((screen.getByRole('button', { name: 'Verify' }) as HTMLButtonElement).disabled).toBe(
		false
	);
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'B edit' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await waitFor(() => expect(update).toHaveBeenCalledOnce());
	expect(update.mock.calls[0]).toEqual([
		second.id,
		expect.objectContaining({ collection_ids: [], expected_operation_revision: 7 })
	]);
});

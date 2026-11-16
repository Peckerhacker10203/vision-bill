import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { api, ApiError } from '$lib/api/client';
import { queryClient } from '$lib/query/client';
import type { ReceiptRow, ReceiptWithDetails } from '$lib/types';
import Page from './+page.svelte';
vi.mock('$app/state', () => ({
	page: { url: new URL('http://localhost/receipts/00000000-0000-4000-8000-000000000001') }
}));
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
	image_path: null
});
beforeEach(() => {
	queryClient.clear();
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
	await screen.findByLabelText('Review comment: total_mismatch');
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
	await screen.findByText('integrity: missing_original');
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

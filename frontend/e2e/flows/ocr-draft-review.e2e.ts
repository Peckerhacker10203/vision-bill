import { expect, test } from '@playwright/test';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { ensureSession, confirmDialog } from '../helpers/ui';
import { awaitSubmission, uploadViaApi } from '../helpers/api';
import type { ReceiptWithDetails } from '../../src/lib/types';

// Uses the real durable direct-vision fallback and existing deterministic fixtures.
// Native OCR unwarped evidence requires Task14's OCR stub; see Task12 report.
test.describe('draft review and versioned receipt editing', () => {
	stubGuard();
	test('partial observations stay nullable until explicitly completed; finalize yields one unverified receipt', async ({
		page,
		context
	}) => {
		test.setTimeout(120_000);
		await ensureSession(context);
		const fixture = await loadFixture('a');
		const uploaded = await uploadViaApi(context, fixture.name, fixture.bytes, fixture.mimeType);
		expect(uploaded.status).toBe(202);
		const id = uploaded.body.id as string;
		const job = await awaitSubmission(context, id);
		expect(job.status).toBe('needs_review');
		await page.goto(`/upload?job=${id}&draft=1`);
		await expect(page.getByLabel('Merchant', { exact: true })).toHaveValue(
			fixture.expected.merchant_name
		);
		const before = await context.request.get(`/api/v1/images/${id}/draft`);
		const draft = await before.json();
		await page.getByLabel('Merchant', { exact: true }).fill('Human correction');
		const patchPromise = page.waitForRequest(
			(r) => r.method() === 'PATCH' && r.url().endsWith(`/images/${id}/draft`)
		);
		await page.getByRole('button', { name: 'Save draft' }).click();
		const patch = await patchPromise;
		expect(patch.postDataJSON()).toEqual({
			expected_revision: draft.revision,
			changes: { merchant_name: 'Human correction' }
		});
		await expect(page.getByRole('button', { name: 'Save draft' })).toBeDisabled();
		const partial = (await (await context.request.get(`/api/v1/images/${id}/draft`)).json())
			.document;
		expect(partial.discount_total).toBeNull();
		expect(partial.total).toBe(draft.document.total);
		await page.getByRole('button', { name: 'Finalize draft' }).click();
		await expect(page.getByRole('alert')).toContainText('discount_total');
		await page.getByLabel('Discount', { exact: true }).fill('0');
		for (const button of await page.getByRole('button', { name: 'Add tags', exact: true }).all())
			await button.click();
		await page.getByRole('button', { name: 'Save draft' }).click();
		await expect(page.getByRole('button', { name: 'Save draft' })).toBeDisabled();
		const finalize = page.waitForResponse(
			(r) => r.request().method() === 'POST' && r.url().endsWith('/draft/finalize')
		);
		await page.getByRole('button', { name: 'Finalize draft' }).click();
		expect((await finalize).status()).toBe(200);
		await expect(page).toHaveURL(/\/receipts\/[a-f0-9-]+/);
		const receiptId = page.url().split('/').pop()!;
		const result = await context.request.get(`/api/v1/receipts/${receiptId}`);
		expect((await result.json()).receipt).toMatchObject({
			verified: false,
			status: 'unverified',
			merchant_name: 'Human correction'
		});
		const finalized = await context.request.get(`/api/v1/images/${id}/draft`);
		expect((await finalized.json()).finalized_receipt_id).toBe(receiptId);
	});
	test('two stale concurrent PUTs cannot overwrite, and a material edit revokes verification', async ({
		page,
		context
	}) => {
		test.setTimeout(120_000);
		await ensureSession(context);
		const fixture = await loadFixture('a');
		const uploaded = await uploadViaApi(context, fixture.name, fixture.bytes, fixture.mimeType);
		const id = uploaded.body.id as string;
		const job = await awaitSubmission(context, id);
		expect(job.status).toBe('needs_review');
		const draft = await (await context.request.get(`/api/v1/images/${id}/draft`)).json();
		const document = {
			...draft.document,
			discount_total: '0',
			line_items: draft.document.line_items.map((line: Record<string, unknown>) => ({
				...line,
				tags: line.tags ?? []
			}))
		};
		const patched = await context.request.patch(`/api/v1/images/${id}/draft`, {
			data: {
				expected_revision: draft.revision,
				changes: {
					...document,
					amount_policy: {
						tax_basis: 'net',
						discount_treatment: 'included_in_lines',
						tip_treatment: 'none',
						subtotal_origin: 'printed',
						tax_information: 'present'
					}
				}
			}
		});
		expect(patched.status()).toBe(200);
		const final = await context.request.post(`/api/v1/images/${id}/draft/finalize`, {
			data: { expected_revision: (await patched.json()).revision, request_key: crypto.randomUUID() }
		});
		expect(final.status()).toBe(200);
		const receiptId = (await final.json()).id;
		await page.goto(`/receipts/${receiptId}`);
		const detail = (await (
			await context.request.get(`/api/v1/receipts/${receiptId}`)
		).json()) as ReceiptWithDetails;
		expect(detail.receipt.findings.filter((f) => f.severity !== 'info')).toEqual([]);
		await page.getByRole('button', { name: 'Verify', exact: true }).click();
		await confirmDialog(page, 'Verify receipt?', 'Verify');
		await expect(page.getByText('Verified', { exact: true })).toBeVisible();
		const verified = (await (
			await context.request.get(`/api/v1/receipts/${receiptId}`)
		).json()) as ReceiptWithDetails;
		const write = {
			...verified.receipt,
			line_items: verified.line_items,
			taxes: verified.taxes,
			collection_ids: verified.collection_ids,
			expected_operation_revision: verified.receipt.operation_revision,
			merchant_name: 'Concurrent edit'
		};
		const results = await Promise.all([
			context.request.put(`/api/v1/receipts/${receiptId}`, { data: write }),
			context.request.put(`/api/v1/receipts/${receiptId}`, { data: write })
		]);
		expect(results.map((r) => r.status()).sort()).toEqual([200, 409]);
		const current = (await (
			await context.request.get(`/api/v1/receipts/${receiptId}`)
		).json()) as ReceiptWithDetails;
		expect(current.receipt.verified).toBe(false);
		expect(current.line_items.map((line) => line.id)).toEqual(
			verified.line_items.map((line) => line.id)
		);
		expect(current.line_items.map((line) => line.position)).toEqual(
			verified.line_items.map((_, index) => index)
		);
		await page.reload();
		await expect(page.getByRole('button', { name: 'Verify', exact: true })).toBeVisible();
	});
});

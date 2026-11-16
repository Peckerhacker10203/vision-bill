import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, expect, it, vi } from 'vitest';
import { session } from '$lib/auth';
import Page from './+page.svelte';
vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));
afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

it('requires unchecked explicit consent and sends the preview revision/hash exactly', async () => {
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
	const calls: { url: string; init?: RequestInit }[] = [];
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string, init?: RequestInit) => {
			calls.push({ url, init });
			if (url === '/api/v1/llm/models') return Response.json([{ id: 'model' }]);
			if (url.endsWith('/capabilities'))
				return Response.json({ ocr_available: false, thinking_control: false });
			if (url.includes('/references/'))
				return Response.json({
					receipt_identity: 'receipt',
					review_revision: 7,
					document_sha256: 'a'.repeat(64),
					scored_fields: ['total'],
					expected: { total: '2.00' },
					sources: [],
					findings: []
				});
			return Response.json({});
		})
	);
	render(Page);
	await screen.findByText('model');
	await fireEvent.input(screen.getByLabelText('Explicit receipt IDs'), {
		target: { value: 'receipt' }
	});
	await fireEvent.click(screen.getByRole('button', { name: 'Preview human references' }));
	const consent = await screen.findByLabelText(
		'I checked these current receipts and originals and explicitly qualify them as benchmark references.'
	);
	expect((consent as HTMLInputElement).checked).toBe(false);
	expect(calls.some((c) => c.init?.method === 'POST')).toBe(false);
	await fireEvent.click(consent);
	await fireEvent.click(screen.getByRole('button', { name: 'Create benchmark' }));
	await fireEvent.click(screen.getByRole('button', { name: 'Confirm and start' }));
	await waitFor(() => expect(calls.filter((c) => c.init?.method === 'POST')).toHaveLength(2));
	const writes = calls.filter((c) => c.init?.method === 'POST');
	expect(writes[0].url).toBe('/api/v1/benchmarks/references/receipt');
	expect(JSON.parse(writes[0].init!.body as string)).toEqual({
		expected_review_revision: 7,
		expected_document_sha256: 'a'.repeat(64)
	});
});

it('loads text formatter models for forced OCR and offers no misleading fallback override', async () => {
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
	const calls: string[] = [];
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string) => {
			calls.push(url);
			if (url.endsWith('/capabilities'))
				return Response.json({ ocr_available: true, ocr_enabled: false, thinking_control: false });
			return Response.json([
				{ id: url.includes('capability=text') ? 'text-only' : 'vision-model' }
			]);
		})
	);
	render(Page);
	await screen.findByText('vision-model');
	await fireEvent.change(screen.getByLabelText('Pipeline'), { target: { value: 'ocr' } });
	await screen.findByText('text-only');
	expect(calls).toContain('/api/v1/llm/models?capability=text');
	expect(screen.queryByLabelText(/Allow direct vision fallback/)).toBeNull();
	await fireEvent.change(screen.getByLabelText('Pipeline'), { target: { value: 'direct_vision' } });
	await screen.findByText('vision-model');
});

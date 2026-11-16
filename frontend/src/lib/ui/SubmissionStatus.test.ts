import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api, ApiError } from '$lib/api/client';
import { queryClient } from '$lib/query/client';
import { queryKeys } from '$lib/query/keys';
import SubmissionStatus from './SubmissionStatus.svelte';

vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
const status = (overrides = {}) => ({
	id: 'job',
	status: 'failed',
	terminal: false,
	generation: 1,
	attempt_count: 1,
	max_attempts: 3,
	retry_at: '2030-01-02T03:04:05Z',
	wait_reason: 'retry_backoff',
	result_code: 'provider_error',
	receipt_id: null,
	draft_available: false,
	source_views: [],
	...overrides
});
afterEach(() => {
	cleanup();
	queryClient.clear();
	vi.restoreAllMocks();
	vi.useRealTimers();
});

describe('SubmissionStatus', () => {
	it.each([
		['failed', 'retry_backoff'],
		['pending', 'ocr_unavailable']
	])('polls nonterminal %s / %s and stops only when terminal', async (state, reason) => {
		vi.useFakeTimers();
		vi.spyOn(api, 'getImage')
			.mockResolvedValueOnce(status({ status: state, wait_reason: reason }) as never)
			.mockResolvedValue(
				status({ terminal: true, status: 'analyzed', receipt_id: 'receipt' }) as never
			);
		render(SubmissionStatus, { id: 'job' });
		await vi.advanceTimersByTimeAsync(50);
		expect(queryClient.getQueryData(queryKeys.image('job'))).toMatchObject({ terminal: false });
		expect(screen.getByText('Attempt 1 of 3')).toBeTruthy();
		expect(screen.getByText(/2030/)).toBeTruthy();
		await vi.advanceTimersByTimeAsync(2200);
		expect(api.getImage).toHaveBeenCalledTimes(2);
		expect(screen.getByRole('link', { name: /Open and review/ }).getAttribute('href')).toBe(
			'/receipts/receipt'
		);
		await vi.advanceTimersByTimeAsync(6000);
		expect(api.getImage).toHaveBeenCalledTimes(2);
	});
	it('offers draft access and authenticated source links for needs_review', async () => {
		vi.spyOn(api, 'getImage').mockResolvedValue(
			status({
				terminal: true,
				status: 'needs_review',
				draft_available: true,
				source_views: [
					{
						view_id: 'view',
						photo_index: 0,
						coordinate_space: 'original',
						url: '/api/v1/images/job/views/view'
					}
				]
			}) as never
		);
		render(SubmissionStatus, { id: 'job' });
		expect((await screen.findByRole('link', { name: /draft/i })).getAttribute('href')).toBe(
			'/upload?job=job&draft=1'
		);
		expect(screen.getByRole('link', { name: /Photo 1/ }).getAttribute('href')).toBe(
			'/api/v1/images/job/views/view'
		);
	});
	it('suppresses duplicate retry clicks and preserves stale-generation detail', async () => {
		vi.spyOn(api, 'getImage').mockResolvedValue(
			status({ terminal: true, retry_at: null, wait_reason: null }) as never
		);
		let reject!: (error: Error) => void;
		const retry = vi.spyOn(api, 'retrySubmission').mockImplementation(
			() =>
				new Promise((_, no) => {
					reject = no;
				})
		);
		render(SubmissionStatus, { id: 'job' });
		const button = await screen.findByRole('button', { name: /Retry analysis/ });
		await fireEvent.click(button);
		await fireEvent.click(button);
		expect(retry).toHaveBeenCalledTimes(1);
		expect(retry.mock.calls[0]).toEqual([
			'job',
			{ expected_generation: 1, request_key: expect.any(String) }
		]);
		reject(new ApiError(409, 'Retry conflict: stale generation'));
		await waitFor(() =>
			expect(screen.getByRole('alert').textContent).toContain('Retry conflict: stale generation')
		);
	});
});

it('reuses retry request identity after an ambiguous retry and adopts the next generation', async () => {
	const getImage = vi
		.spyOn(api, 'getImage')
		.mockResolvedValue(status({ terminal: true, retry_at: null, wait_reason: null }) as never);
	const retry = vi
		.spyOn(api, 'retrySubmission')
		.mockRejectedValueOnce(new ApiError(0, 'Network error'))
		.mockResolvedValue(
			status({
				status: 'pending',
				generation: 2,
				wait_reason: 'ocr_unavailable',
				retry_at: null
			}) as never
		);
	render(SubmissionStatus, { id: 'job' });
	await fireEvent.click(await screen.findByRole('button', { name: 'Retry analysis' }));
	await screen.findByRole('alert');
	getImage.mockResolvedValue(
		status({
			status: 'pending',
			generation: 2,
			wait_reason: 'ocr_unavailable',
			retry_at: null
		}) as never
	);
	await fireEvent.click(screen.getByRole('button', { name: 'Retry analysis' }));
	await waitFor(() => expect(retry).toHaveBeenCalledTimes(2));
	expect(retry.mock.calls[0]).toEqual(retry.mock.calls[1]);
	await waitFor(() =>
		expect(queryClient.getQueryData(queryKeys.image('job'))).toMatchObject({ generation: 2 })
	);
});

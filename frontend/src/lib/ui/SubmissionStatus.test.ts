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

function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (error: Error) => void;
	const promise = new Promise<T>((yes, no) => {
		resolve = yes;
		reject = no;
	});
	return { promise, resolve, reject };
}

it('keeps B and its pending GET untouched when A retry completes after navigation', async () => {
	const retryA = deferred<ReturnType<typeof status>>();
	const getB = deferred<ReturnType<typeof status>>();
	vi.spyOn(api, 'getImage').mockImplementation((id) =>
		id === 'a'
			? Promise.resolve(status({ id: 'a', terminal: true }) as never)
			: (getB.promise as never)
	);
	vi.spyOn(api, 'retrySubmission').mockReturnValue(retryA.promise as never);
	const cancel = vi.spyOn(queryClient, 'cancelQueries');
	const { rerender } = render(SubmissionStatus, { id: 'a' });
	await fireEvent.click(await screen.findByRole('button', { name: 'Retry analysis' }));
	await rerender({ id: 'b' });
	await waitFor(() => expect(api.getImage).toHaveBeenCalledWith('b'));
	retryA.resolve(
		status({ id: 'a', generation: 2, status: 'analyzed', terminal: true, receipt_id: 'a-receipt' })
	);
	await retryA.promise;
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(queryClient.getQueryData(queryKeys.image('b'))).toBeUndefined();
	expect(
		cancel.mock.calls.some(
			([filters]) => JSON.stringify(filters?.queryKey) === JSON.stringify(queryKeys.image('b'))
		)
	).toBe(false);
	getB.resolve(status({ id: 'b', terminal: true, status: 'analyzed', receipt_id: 'b-receipt' }));
	expect(
		(await screen.findByRole('link', { name: 'Open and review receipt' })).getAttribute('href')
	).toBe('/receipts/b-receipt');
});

it.each([0, 409])('ignores A late %s error and leaves B retry in progress', async (code) => {
	const retryA = deferred<ReturnType<typeof status>>();
	const retryB = deferred<ReturnType<typeof status>>();
	vi.spyOn(api, 'getImage').mockImplementation((id) =>
		Promise.resolve(status({ id, terminal: true }) as never)
	);
	const retry = vi
		.spyOn(api, 'retrySubmission')
		.mockImplementation((id) => (id === 'a' ? retryA.promise : retryB.promise) as never);
	const { rerender } = render(SubmissionStatus, { id: 'a' });
	await fireEvent.click(await screen.findByRole('button', { name: 'Retry analysis' }));
	await rerender({ id: 'b' });
	const buttonB = await screen.findByRole('button', { name: 'Retry analysis' });
	await fireEvent.click(buttonB);
	const readsBefore = vi.mocked(api.getImage).mock.calls.length;
	retryA.reject(new ApiError(code, 'A late error'));
	await retryA.promise.catch(() => {});
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(screen.queryByRole('alert')).toBeNull();
	expect((buttonB as HTMLButtonElement).disabled).toBe(true);
	expect(retry).toHaveBeenCalledTimes(2);
	expect(api.getImage).toHaveBeenCalledTimes(readsBefore);
	retryB.reject(new ApiError(0, 'B current error'));
	expect((await screen.findByRole('alert')).textContent).toContain('B current error');
});

it.each([false, true])('ignores retry completion after unmount (conflict=%s)', async (conflict) => {
	const pending = deferred<ReturnType<typeof status>>();
	const initial = status({ terminal: true });
	vi.spyOn(api, 'getImage').mockResolvedValue(initial as never);
	vi.spyOn(api, 'retrySubmission').mockReturnValue(pending.promise as never);
	const { unmount } = render(SubmissionStatus, { id: 'job' });
	await fireEvent.click(await screen.findByRole('button', { name: 'Retry analysis' }));
	unmount();
	const cancel = vi.spyOn(queryClient, 'cancelQueries');
	const readsBefore = vi.mocked(api.getImage).mock.calls.length;
	if (conflict) pending.reject(new ApiError(409, 'Unmounted conflict'));
	else pending.resolve(status({ generation: 2 }));
	await pending.promise.catch(() => {});
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(cancel).not.toHaveBeenCalled();
	expect(api.getImage).toHaveBeenCalledTimes(readsBefore);
	expect(queryClient.getQueryData(queryKeys.image('job'))).toEqual(initial);
});

it.each([
	{ generation: 3, status: 'pending', terminal: false },
	{ generation: 2, status: 'analyzed', terminal: true, receipt_id: 'current-receipt' }
])('does not regress a newer cached generation/state: %j', async (newer) => {
	const pending = deferred<ReturnType<typeof status>>();
	const getImage = vi.spyOn(api, 'getImage').mockResolvedValue(status({ terminal: true }) as never);
	vi.spyOn(api, 'retrySubmission').mockReturnValue(pending.promise as never);
	render(SubmissionStatus, { id: 'job' });
	await fireEvent.click(await screen.findByRole('button', { name: 'Retry analysis' }));
	const latest = status(newer);
	getImage.mockResolvedValue(latest as never);
	queryClient.setQueryData(queryKeys.image('job'), latest);
	const observed: unknown[] = [];
	const unsubscribe = queryClient.getQueryCache().subscribe((event) => {
		if (event.query.queryKey[1] === 'job') observed.push(event.query.state.data);
	});
	pending.resolve(status({ generation: 2, status: 'pending' }));
	await pending.promise;
	await new Promise((resolve) => setTimeout(resolve, 0));
	unsubscribe();
	expect(observed.every((value) => JSON.stringify(value) === JSON.stringify(latest))).toBe(true);
	expect(queryClient.getQueryData(queryKeys.image('job'))).toEqual(latest);
});

it('rechecks request ownership after awaiting query cancellation', async () => {
	const pending = deferred<ReturnType<typeof status>>();
	const cancellation = deferred<void>();
	vi.spyOn(api, 'getImage').mockImplementation((id) =>
		Promise.resolve(status({ id, terminal: true }) as never)
	);
	vi.spyOn(api, 'retrySubmission').mockReturnValue(pending.promise as never);
	const cancel = vi.spyOn(queryClient, 'cancelQueries').mockReturnValue(cancellation.promise);
	const { rerender } = render(SubmissionStatus, { id: 'a' });
	await fireEvent.click(await screen.findByRole('button', { name: 'Retry analysis' }));
	pending.resolve(status({ id: 'a', generation: 2 }));
	await waitFor(() => expect(cancel).toHaveBeenCalledOnce());
	await rerender({ id: 'b' });
	await waitFor(() =>
		expect(queryClient.getQueryData(queryKeys.image('b'))).toMatchObject({ id: 'b' })
	);
	cancellation.resolve();
	await cancellation.promise;
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(queryClient.getQueryData(queryKeys.image('b'))).toMatchObject({ id: 'b', generation: 1 });
	expect(queryClient.getQueryData(queryKeys.image('a'))).toMatchObject({ id: 'a', generation: 1 });
	expect(
		(screen.getByRole('button', { name: 'Retry analysis' }) as HTMLButtonElement).disabled
	).toBe(false);
});

it('ignores the first A retry after returning to A and starting another request', async () => {
	const first = deferred<ReturnType<typeof status>>();
	const second = deferred<ReturnType<typeof status>>();
	vi.spyOn(api, 'getImage').mockImplementation((id) =>
		Promise.resolve(status({ id, terminal: true }) as never)
	);
	const retry = vi
		.spyOn(api, 'retrySubmission')
		.mockReturnValueOnce(first.promise as never)
		.mockReturnValueOnce(second.promise as never);
	const { rerender } = render(SubmissionStatus, { id: 'a' });
	await fireEvent.click(await screen.findByRole('button', { name: 'Retry analysis' }));
	await rerender({ id: 'b' });
	await waitFor(() => expect(api.getImage).toHaveBeenCalledWith('b'));
	await rerender({ id: 'a' });
	const current = await screen.findByRole('button', { name: 'Retry analysis' });
	await fireEvent.click(current);
	expect(retry).toHaveBeenCalledTimes(2);
	first.reject(new ApiError(409, 'Earlier A conflict'));
	await first.promise.catch(() => {});
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(screen.queryByRole('alert')).toBeNull();
	expect((current as HTMLButtonElement).disabled).toBe(true);
	second.reject(new ApiError(0, 'Current A error'));
	expect((await screen.findByRole('alert')).textContent).toContain('Current A error');
});

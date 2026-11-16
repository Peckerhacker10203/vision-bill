<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
	import { resolve } from '$app/paths';
	import { api, ApiError } from '$lib/api/client';
	import { requestKey } from '$lib/api/upload';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import { t } from '$lib/i18n';

	let { id }: { id: string } = $props();
	let retrying = $state(false);
	let retryError = $state('');
	let retryRequest: { id: string; expected_generation: number; request_key: string } | null = null;
	const job = createQuery(
		() => ({
			queryKey: queryKeys.image(id),
			queryFn: () => api.getImage(id),
			retry: false,
			staleTime: 0,
			refetchInterval: (query) => (query.state.data?.terminal ? false : 2000)
		}),
		() => queryClient
	);

	async function retry() {
		if (retrying || !job.data?.terminal || job.data.status === 'analyzed') return;
		retrying = true;
		retryError = '';
		try {
			if (retryRequest?.id !== id || retryRequest.expected_generation !== job.data.generation) {
				retryRequest = { id, expected_generation: job.data.generation, request_key: requestKey() };
			}
			const result = await api.retrySubmission(id, {
				expected_generation: retryRequest.expected_generation,
				request_key: retryRequest.request_key
			});
			await queryClient.cancelQueries({ queryKey: queryKeys.image(id) });
			queryClient.setQueryData(queryKeys.image(id), result);
			retryRequest = null;
			void queryClient.invalidateQueries({ queryKey: ['images'] });
		} catch (error) {
			retryError = error instanceof ApiError ? `${error.status}: ${error.detail}` : String(error);
			// A conflict needs fresh generation state, while its detail remains visible.
			if (error instanceof ApiError && error.status === 409) void job.refetch();
		} finally {
			retrying = false;
		}
	}
</script>

<div class="mt-4 rounded-xl border border-outline-variant bg-surface-container-low p-5">
	{#if job.error}
		<p role="alert">{$t('submission.loadError')} {job.error.message}</p>
		<button type="button" class="mt-3 text-primary underline" onclick={() => job.refetch()}
			>{$t('submission.refresh')}</button
		>
	{:else if !job.data}
		<p role="status">{$t('common.loading')}</p>
	{:else}
		<p class="text-lg font-medium" role="status" aria-live="polite">
			{$t(`submission.${job.data.status}`)}
		</p>
		<p class="mt-2 text-sm">
			{$t('submission.attempt', {
				values: { count: job.data.attempt_count, max: job.data.max_attempts }
			})}
		</p>
		{#if !job.data.terminal}
			<p class="mt-2 text-sm text-on-surface-variant">{$t('submission.returnLater')}</p>
			{#if job.data.retry_at}
				<p class="mt-2 text-sm">
					{$t('submission.retryAt', {
						values: { time: new Date(job.data.retry_at).toLocaleString() }
					})}
				</p>
			{/if}
			{#if job.data.wait_reason}<p class="mt-2 text-sm">
					{$t('submission.waitReason')}: {job.data.wait_reason}
				</p>{/if}
		{/if}
		{#if job.data.result_code}<p class="mt-2 text-sm">
				{$t('submission.resultCode')}: {job.data.result_code}
			</p>{/if}
		{#if job.data.error}<p class="mt-2 text-sm text-error">{job.data.error}</p>{/if}
		{#if job.data.receipt_id}
			<a
				class="mt-4 inline-block rounded-lg bg-primary px-4 py-2 text-on-primary"
				href={resolve(`/receipts/${job.data.receipt_id}`)}>{$t('submission.openReceipt')}</a
			>
		{/if}
		{#if job.data.draft_available}
			<a
				class="mt-4 inline-block text-primary underline"
				href={resolve(`/upload?job=${id}&draft=1`)}>{$t('submission.openDraft')}</a
			>
		{/if}
		{#if job.data.terminal && job.data.status !== 'analyzed'}
			<button
				type="button"
				class="mt-4 rounded-lg bg-primary px-4 py-2 text-on-primary disabled:opacity-50"
				disabled={retrying}
				onclick={retry}>{$t('submission.retry')}</button
			>
		{/if}
		{#if job.data.source_views?.length}
			<ul class="mt-4 flex flex-wrap gap-3">
				{#each job.data.source_views as view (view.view_id)}
					<li>
						<!-- Authenticated artifact URLs are supplied by the API, not SPA routes. -->
						<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -->
						<a class="text-primary underline" href={view.url} target="_blank" rel="noreferrer"
							>{$t('submission.photo', { values: { number: view.photo_index + 1 } })} ({view.coordinate_space})</a
						>
					</li>
				{/each}
			</ul>
		{/if}
		<p class="mt-4 break-all text-xs text-on-surface-variant">{$t('submission.reference')}: {id}</p>
	{/if}
	{#if retryError}<p class="mt-3 text-sm text-error" role="alert">{retryError}</p>{/if}
	<a class="mt-4 inline-block text-primary underline" href={resolve('/upload')}
		>{$t('submission.newUpload')}</a
	>
</div>

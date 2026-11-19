<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
	import { api } from '$lib/api/client';
	import { t } from '$lib/i18n';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import { captureSessionActor } from '$lib/session-lifetime';
	import { sessionViewGuard } from '$lib/session-view';

	const actor = captureSessionActor();
	const isCurrentView = sessionViewGuard();
	const queryKey = [...queryKeys.dedicatedOcr(), actor?.generation];
	const status = createQuery(
		() => ({
			queryKey,
			queryFn: () => api.getDedicatedOcr(actor),
			staleTime: 0,
			refetchOnWindowFocus: true,
			refetchInterval: (query) => {
				const state = query.state.data?.state;
				return state === 'requested' || state === 'installing' ? 2000 : false;
			}
		}),
		() => queryClient
	);

	const reasons = new Set([
		'gpu_profile_required',
		'ocr_config_managed_externally',
		'llm_lifecycle_required',
		'llm_quiescence_required',
		'llm_request_recovery_required',
		'ocr_handoff_confirmation_required',
		'ocr_runtime_unavailable',
		'ocr_download_failed'
	]);
	const activationState = $derived(status.data?.state);
	const reason = $derived(status.data?.reason);
	const runtimeUnavailable = $derived(reason === 'ocr_runtime_unavailable');
	const message = $derived(
		reason && reasons.has(reason)
			? `settings.dedicatedOcr.reasons.${reason}`
			: `settings.dedicatedOcr.states.${activationState}`
	);
	const canRequest = $derived(
		!runtimeUnavailable &&
			(activationState === 'disabled' ||
				activationState === 'failed' ||
				activationState === 'action_required')
	);
	let confirmHandoff = $state(false);
	let submitting = $state(false);
	let requestFailed = $state(false);

	$effect(() => {
		if (activationState === 'active' && isCurrentView()) {
			void queryClient.invalidateQueries({ queryKey: queryKeys.uiConfig() });
		}
	});

	async function requestSetup() {
		const previous = status.data;
		if (!previous || !canRequest || !confirmHandoff || submitting || !isCurrentView()) return;
		submitting = true;
		requestFailed = false;
		try {
			const request = api.requestDedicatedOcr(actor, confirmHandoff);
			confirmHandoff = false;
			const result = await request;
			if (!isCurrentView()) return;
			await queryClient.cancelQueries({ queryKey });
			if (isCurrentView()) queryClient.setQueryData(queryKey, result);
		} catch {
			if (!isCurrentView()) return;
			const refreshed = await status.refetch();
			if (isCurrentView()) {
				requestFailed =
					refreshed.isError ||
					(refreshed.data?.state === previous.state && refreshed.data.reason === previous.reason);
			}
		} finally {
			if (isCurrentView()) submitting = false;
		}
	}
</script>

<section
	class="mt-6 max-w-2xl space-y-3 rounded-xl border border-outline-variant bg-surface-container-low p-5"
	aria-labelledby="dedicated-ocr-title"
>
	<h2 class="font-semibold" id="dedicated-ocr-title">{$t('settings.dedicatedOcr.title')}</h2>
	{#if activationState}
		<p class="text-sm text-on-surface-variant" role="status">{$t(message)}</p>
	{:else if status.isPending}
		<p class="text-sm text-on-surface-variant" role="status">{$t('common.loading')}</p>
	{/if}
	{#if status.isError}
		<p class="text-sm text-error" role="alert">{$t('settings.dedicatedOcr.loadError')}</p>
	{/if}
	{#if status.isError || runtimeUnavailable}
		<button
			type="button"
			class="rounded-lg border border-outline-variant px-4 py-2.5 text-sm font-medium disabled:opacity-50"
			disabled={status.isFetching || submitting}
			onclick={() => void status.refetch()}
		>
			{$t('settings.dedicatedOcr.retryStatus')}
		</button>
	{/if}
	{#if requestFailed}
		<p class="text-sm text-error" role="alert">{$t('settings.dedicatedOcr.requestError')}</p>
	{/if}
	{#if canRequest && !status.isError}
		<label class="flex items-start gap-2 text-sm">
			<input
				type="checkbox"
				class="mt-0.5 size-4 shrink-0"
				bind:checked={confirmHandoff}
				disabled={submitting}
			/>
			<span>{$t('settings.dedicatedOcr.confirmHandoff')}</span>
		</label>
		<button
			type="button"
			class="rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-on-primary disabled:opacity-50"
			disabled={submitting || !confirmHandoff}
			onclick={requestSetup}
		>
			{submitting
				? $t('settings.dedicatedOcr.submitting')
				: activationState === 'disabled'
					? $t('settings.dedicatedOcr.enable')
					: $t('settings.dedicatedOcr.retry')}
		</button>
	{/if}
</section>

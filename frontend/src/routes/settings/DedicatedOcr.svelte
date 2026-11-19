<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
	import { api, ApiError } from '$lib/api/client';
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
		'model_request_unsettled',
		'model_handoff_model_not_adopted',
		'ocr_setup_pending',
		'ocr_activation_state_invalid',
		'ocr_config_managed_externally',
		'llm_lifecycle_required',
		'llm_quiescence_required',
		'llm_request_recovery_required',
		'ocr_handoff_confirmation_required',
		'ocr_runtime_unavailable',
		'ocr_download_failed'
	]);
	const installed = $derived(status.data?.installed ?? false);
	const activationState = $derived(status.data?.state);
	const selected = $derived(status.data?.selected ?? activationState === 'active');
	const reason = $derived(status.data?.reason);
	const runtimeUnavailable = $derived(reason === 'ocr_runtime_unavailable');
	const message = $derived(
		reason && reasons.has(reason)
			? `settings.dedicatedOcr.reasons.${reason}`
			: installed && activationState === 'disabled'
				? 'settings.dedicatedOcr.installedOff'
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
	let requestReason = $state<string | null>(null);
	const busy = $derived(
		submitting || activationState === 'requested' || activationState === 'installing'
	);

	$effect(() => {
		if ((activationState === 'active' || activationState === 'disabled') && isCurrentView()) {
			void queryClient.invalidateQueries({ queryKey: queryKeys.uiConfig() });
			void queryClient.invalidateQueries({ queryKey: ['models'] });
		}
	});

	async function requestSetup() {
		const previous = status.data;
		const enabled = !selected;
		if (!previous || busy || status.isError || !isCurrentView()) return;
		if (enabled && (!canRequest || (!installed && !confirmHandoff))) return;
		submitting = true;
		requestFailed = false;
		requestReason = null;
		try {
			const request = api.requestDedicatedOcr(actor, confirmHandoff, enabled);
			confirmHandoff = false;
			const result = await request;
			if (!isCurrentView()) return;
			await queryClient.cancelQueries({ queryKey });
			if (isCurrentView()) queryClient.setQueryData(queryKey, result);
		} catch (error) {
			if (!isCurrentView()) return;
			const refreshed = await status.refetch();
			if (isCurrentView()) {
				requestReason =
					error instanceof ApiError &&
					error.status === 409 &&
					reasons.has(error.detail) &&
					error.detail !== refreshed.data?.reason
						? error.detail
						: null;
				requestFailed =
					!requestReason &&
					(refreshed.isError ||
						(refreshed.data?.state === previous.state &&
							refreshed.data.reason === previous.reason));
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
	{#if requestReason}
		<p class="text-sm text-error" role="alert">
			{$t(`settings.dedicatedOcr.reasons.${requestReason}`)}
		</p>
	{:else if requestFailed}
		<p class="text-sm text-error" role="alert">{$t('settings.dedicatedOcr.requestError')}</p>
	{/if}
	<button
		type="button"
		role="switch"
		aria-label={$t('settings.dedicatedOcr.title')}
		aria-checked={selected}
		aria-busy={busy}
		class="inline-flex items-center gap-3 rounded-full border border-outline-variant px-4 py-2.5 text-sm font-medium disabled:opacity-50"
		disabled={busy ||
			status.isError ||
			(!selected && (!canRequest || (!installed && !confirmHandoff)))}
		onclick={requestSetup}
	>
		<span
			aria-hidden="true"
			class:bg-action={selected}
			class="flex h-6 w-11 items-center rounded-full bg-outline-variant p-1"
		>
			<span
				class:translate-x-5={selected}
				class="size-4 rounded-full bg-surface transition-transform"
			></span>
		</span>
		{$t('settings.dedicatedOcr.title')}
	</button>
	{#if canRequest && !installed && !status.isError}
		<label class="flex items-start gap-2 text-sm">
			<input
				type="checkbox"
				class="mt-0.5 size-4 shrink-0"
				bind:checked={confirmHandoff}
				disabled={busy}
			/>
			<span>{$t('settings.dedicatedOcr.confirmHandoff')}</span>
		</label>
	{/if}
	<p class="text-sm text-on-surface-variant">{$t('settings.dedicatedOcr.prerequisites')}</p>
	<a
		class="text-sm underline"
		href="https://github.com/tendstead/vision-bill/blob/main/paddleocr-service/README.md"
		target="_blank"
		rel="noreferrer">{$t('settings.dedicatedOcr.setupGuide')}</a
	>
</section>

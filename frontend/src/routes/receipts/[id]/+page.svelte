<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import { t, translate } from '$lib/i18n';
	import { api, ApiError } from '$lib/api/client';
	import { captureSessionActor, isCurrentSessionActor } from '$lib/session-lifetime';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import { snackbar } from '$lib/ui/snackbar.svelte';
	import ConfirmDialog from '$lib/ui/ConfirmDialog.svelte';
	import Icon from '$lib/ui/Icon.svelte';
	import ReceiptEditor from '$lib/ui/ReceiptEditor.svelte';
	import ReceiptPhotos from '$lib/ui/ReceiptPhotos.svelte';
	import type {
		ReceiptWrite,
		ReceiptRow,
		ReviewFinding,
		ReviewPreview,
		ReviewRequest,
		SourceRef
	} from '$lib/types';
	import ReviewFindings from '$lib/ui/ReviewFindings.svelte';
	import ExtractionSourceViewer from '$lib/ui/ExtractionSourceViewer.svelte';

	let id = $derived(page.url.pathname.split('/').pop() ?? '');
	const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
	let verifyOpen = $state(false);
	let previewing = $state(false);
	let humanPreview = $state<ReviewPreview | null>(null);
	let humanAcknowledgements = $state<ReviewRequest['acknowledgements']>([]);
	let verifying = $state(false);
	let saving = $state(false);
	let editorRevision = $state(0);
	let savedRevision = $state(0);
	let editorDirty = $state(false);
	let childPending = $state(false);
	let acknowledgements = $state<{ finding_id: string; comment: string }[]>([]);
	let acknowledgementReset = $state(0);
	let reviewMessage = $state('');
	let needsReload = $state(false);
	let selectedSources = $state<SourceRef[]>([]);
	const actor = captureSessionActor();
	let activeView: { id: string } | null = null;
	const isCurrentView = (view: typeof activeView) =>
		view !== null && activeView === view && view.id === id && isCurrentSessionActor(actor);
	$effect(() => {
		const view = { id };
		activeView = view;
		editorRevision = 0;
		savedRevision = 0;
		acknowledgementReset = 0;
		saving = false;
		verifying = false;
		verifyOpen = false;
		previewing = false;
		humanPreview = null;
		humanAcknowledgements = [];
		selectedSources = [];
		editorDirty = false;
		childPending = false;
		acknowledgements = [];
		reviewMessage = '';
		needsReload = false;
		return () => {
			if (activeView === view) activeView = null;
		};
	});

	const detail = createQuery(
		() => ({
			queryKey: queryKeys.receipt(id),
			queryFn: () => api.getReceipt(id),
			enabled: uuidPattern.test(id)
		}),
		() => queryClient
	);

	let notFound = $derived(
		!uuidPattern.test(id) ||
			(detail.status === 'error' && detail.error instanceof ApiError && detail.error.status === 404)
	);
	let data = $derived(detail.data ?? null);
	let enrichment = $derived(data?.classification_enrichment);
	const sourceJob = createQuery(
		() => ({
			queryKey: queryKeys.image(data?.receipt.image_id ?? ''),
			queryFn: () => api.getImage(data!.receipt.image_id!),
			enabled: !!data?.receipt.image_id
		}),
		() => queryClient
	);
	let canVerify = $derived(data !== null && data.receipt.status === 'unverified');
	let canReviewAsHuman = $derived(
		data?.receipt.verified &&
			data.receipt.status === 'verified' &&
			data.receipt.verification_source !== 'human'
	);
	let reviewBlocked = $derived(editorDirty || saving || verifying || needsReload || childPending);

	function findingsReady(findings: ReviewFinding[], comments: ReviewRequest['acknowledgements']) {
		return findings.every(
			(finding) =>
				finding.severity === 'info' ||
				(finding.severity === 'review' &&
					comments.some((comment) => comment.finding_id === finding.id && comment.comment.trim()))
		);
	}
	let reviewReady = $derived(!!data && findingsReady(data.receipt.findings, acknowledgements));
	let humanReviewReady = $derived(
		!!humanPreview && findingsReady(humanPreview.findings, humanAcknowledgements)
	);
	function reviewBody(
		receipt: Pick<ReceiptRow, 'operation_revision' | 'review_revision'>,
		comments = acknowledgements
	) {
		return {
			expected_operation_revision: receipt.operation_revision,
			expected_review_revision: receipt.review_revision,
			acknowledgements: comments
		};
	}
	async function refresh(requestId = id, view = activeView): Promise<void> {
		if (!isCurrentView(view)) return;
		const fresh = await api.getReceipt(requestId);
		if (!isCurrentView(view)) return;
		queryClient.setQueryData(queryKeys.receipt(requestId), fresh);
		editorRevision++;
		needsReload = false;
		void queryClient.invalidateQueries({ queryKey: ['statistics'] });
	}
	async function discardReload() {
		const requestId = id;
		const view = activeView;
		if (!isCurrentView(view)) return;
		saving = true;
		try {
			await refresh(requestId, view);
			if (!isCurrentView(view)) return;
			reviewMessage = '';
		} catch {
			if (!isCurrentView(view)) return;
			reviewMessage = translate('receipt.saveFailed');
		} finally {
			if (isCurrentView(view)) saving = false;
		}
	}
	async function save(write: ReceiptWrite, verify: boolean) {
		if (saving || needsReload || childPending) return;
		const requestId = id;
		const view = activeView;
		if (!isCurrentView(view)) return;
		saving = true;
		reviewMessage = '';
		let saved: ReceiptRow | null = null;
		try {
			saved = await api.updateReceipt(requestId, write);
			if (!isCurrentView(view)) return;
			savedRevision++;
			acknowledgementReset++;
			acknowledgements = [];
			needsReload = true;
			queryClient.setQueryData<import('$lib/types').ReceiptWithDetails>(
				queryKeys.receipt(requestId),
				(old) => (old ? { ...old, receipt: saved! } : old)
			);
			snackbar.notify('success', translate('receipt.saved'));
			await refresh(requestId, view);
			if (!isCurrentView(view)) return;
			if (verify) {
				if (saved.findings.some((f) => f.severity !== 'info')) {
					reviewMessage = translate('review.savedReview');
					return;
				}
				await api.verifyReceipt(requestId, reviewBody(saved, []));
				if (!isCurrentView(view)) return;
				snackbar.notify('success', translate('receipt.verifySuccess'));
				await refresh(requestId, view);
			}
		} catch (error) {
			if (!isCurrentView(view)) return;
			reviewMessage =
				error instanceof ApiError && error.status === 409
					? translate('review.conflict')
					: translate(saved ? 'receipt.verifyFailed' : 'receipt.saveFailed');
			snackbar.notify('error', reviewMessage);
		} finally {
			if (isCurrentView(view)) saving = false;
		}
	}
	function saveOnly(write: ReceiptWrite) {
		return save(write, false);
	}
	function saveAndVerify(write: ReceiptWrite) {
		return save(write, true);
	}
	function closeReview() {
		verifyOpen = false;
		humanPreview = null;
		humanAcknowledgements = [];
	}
	async function openHumanReview() {
		if (!canReviewAsHuman || previewing || reviewBlocked) return;
		const requestId = id;
		const view = activeView;
		if (!isCurrentView(view)) return;
		previewing = true;
		reviewMessage = '';
		try {
			const preview = await api.previewHumanReview(requestId);
			if (!isCurrentView(view)) return;
			if (!matchesReview(preview)) {
				reviewMessage = translate('review.conflict');
				return;
			}
			if (!reviewBlocked) {
				humanPreview = preview;
				verifyOpen = true;
			}
		} catch (error) {
			if (!isCurrentView(view)) return;
			reviewMessage =
				error instanceof ApiError && error.status === 409
					? translate('review.conflict')
					: translate('receipt.verifyFailed');
		} finally {
			if (isCurrentView(view)) previewing = false;
		}
	}
	function matchesReview(preview: ReviewPreview) {
		return (
			data?.receipt.operation_revision === preview.operation_revision &&
			data.receipt.review_revision === preview.review_revision
		);
	}
	async function doVerify() {
		const requestId = id;
		const view = activeView;
		if (!isCurrentView(view) || !data || reviewBlocked) return;
		const preview = humanPreview;
		if (preview && !matchesReview(preview)) {
			closeReview();
			reviewMessage = translate('review.conflict');
			return;
		}
		if (preview ? !humanReviewReady : !canVerify || !reviewReady) return;
		const request = reviewBody(preview ?? data.receipt, [
			...(preview ? humanAcknowledgements : acknowledgements)
		]);
		verifying = true;
		try {
			await api.verifyReceipt(requestId, request);
			if (!isCurrentView(view)) return;
			snackbar.notify('success', translate('receipt.verifySuccess'));
			closeReview();
			await refresh(requestId, view);
		} catch (error) {
			if (!isCurrentView(view)) return;
			if (preview) closeReview();
			reviewMessage =
				error instanceof ApiError && error.status === 409
					? translate('review.conflict')
					: translate('receipt.verifyFailed');
			snackbar.notify('error', reviewMessage);
		} finally {
			if (isCurrentView(view)) verifying = false;
		}
	}
</script>

<svelte:head>
	<title>{$t('pages.receiptDetail.title', { values: { id } })} – {$t('app.name')}</title>
</svelte:head>

{#if detail.isLoading}
	<p class="text-sm text-on-surface-variant">{$t('common.loading')}</p>
{:else if notFound}
	<div class="rounded-xl border border-outline-variant bg-surface-container-low p-4 text-sm">
		<p class="font-medium">{$t('receipt.notFoundTitle')}</p>
		<p class="mt-1 text-on-surface-variant">{$t('receipt.notFoundBody')}</p>
		<a
			href={resolve('/receipts')}
			class="mt-3 inline-block rounded-lg bg-primary px-4 py-2 text-on-primary hover:opacity-90"
		>
			{$t('common.back')}
		</a>
	</div>
{:else if detail.error}
	<div
		class="rounded-xl border border-outline-variant bg-surface-container-low p-4 text-sm text-on-surface-variant"
	>
		{$t('common.error')}
	</div>
{:else if data}
	<section class="mx-auto min-w-0 max-w-6xl">
		<div class="flex items-center gap-3">
			<a
				href={resolve('/receipts')}
				class="rounded-lg p-2 text-on-surface-variant hover:bg-surface-container"
				aria-label={$t('common.back')}
			>
				<Icon icon="close" />
			</a>
			<h1 class="flex-1 truncate text-xl font-semibold">
				{data.receipt.merchant_name || $t('receipts.unknownVendor')}
			</h1>
			{#if canVerify || canReviewAsHuman}
				<button
					type="button"
					class="flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-on-primary hover:opacity-90 disabled:opacity-50"
					disabled={reviewBlocked || previewing || (canVerify && !reviewReady)}
					onclick={canReviewAsHuman ? openHumanReview : () => (verifyOpen = true)}
				>
					<Icon icon="check" />
					{$t(canReviewAsHuman ? 'receipt.humanReview' : 'receipt.verify')}
				</button>
			{/if}
			{#if !canVerify}
				<span
					class="rounded-full bg-primary-container px-3 py-1 text-xs font-medium text-on-primary-container"
				>
					{$t('receipt.verified')}
				</span>
			{/if}
		</div>

		{#if reviewMessage}<p class="my-3 text-error" role="alert">{reviewMessage}</p>{/if}
		{#if enrichment?.state === 'unavailable' || (enrichment?.state === 'skipped' && enrichment.reason_code === 'classification_budget_exhausted' && enrichment.missing_fields.length > 0)}
			<p class="my-3 text-sm text-on-surface-variant">{$t('receipt.optionalLabelsUnavailable')}</p>
		{/if}
		{#if reviewMessage || needsReload}<button
				type="button"
				disabled={saving || verifying}
				onclick={discardReload}
				class="my-3 text-primary underline">{$t('review.discardReload')}</button
			>{/if}
		<ReviewFindings
			findings={data.receipt.findings}
			revision={`${id}:${data.receipt.review_revision}:${acknowledgementReset}`}
			editable={canVerify && !editorDirty && !saving && !needsReload}
			onChange={(value) => (acknowledgements = value)}
			onSource={(sources) => (selectedSources = sources)}
		/>
		{#if selectedSources.length}<ExtractionSourceViewer
				views={sourceJob.data?.source_views ?? []}
				sources={selectedSources}
			/>{/if}
		<div class="mt-6 min-w-0 lg:grid lg:grid-cols-[minmax(20rem,0.8fr)_minmax(0,1.2fr)] lg:gap-8">
			{#if data.receipt.image_id !== null}
				<aside class="mb-6 min-w-0 lg:mb-0">
					<div class="min-w-0 lg:sticky lg:top-20">
						<ReceiptPhotos
							imageId={data.receipt.image_id}
							thumbnailPath={data.receipt.thumbnail_path ?? null}
							alt={data.receipt.merchant_name || $t('receipts.unknownVendor')}
						/>
					</div>
				</aside>
			{/if}

			<div class="min-w-0 {data.receipt.image_id === null ? 'lg:col-span-2' : ''}">
				<ReceiptEditor
					receipt={data.receipt}
					revision={editorRevision}
					{savedRevision}
					lineItems={data.line_items}
					taxes={data.taxes}
					collectionIds={data.collection_ids}
					busy={saving || verifying || needsReload}
					onSave={saveOnly}
					onSaveAndVerify={canVerify ? saveAndVerify : undefined}
					onDirtyChange={(dirty) => (editorDirty = dirty)}
					onSource={(sources) => (selectedSources = sources)}
					onPendingChange={(pending) => (childPending = pending)}
				/>
			</div>
		</div>
	</section>

	<ConfirmDialog
		open={verifyOpen}
		title={$t(humanPreview ? 'receipt.humanReviewTitle' : 'receipt.verifyTitle')}
		body={$t(humanPreview ? 'receipt.humanReviewBody' : 'receipt.verifyBody')}
		confirmLabel={$t(humanPreview ? 'receipt.humanReview' : 'receipt.verify')}
		busy={verifying}
		confirmDisabled={reviewBlocked || (humanPreview ? !humanReviewReady : !reviewReady)}
		onConfirm={doVerify}
		onCancel={closeReview}
	>
		{#if humanPreview && humanPreview.findings.length}
			<ReviewFindings
				findings={humanPreview.findings}
				revision={humanPreview.operation_revision}
				onChange={(value) => (humanAcknowledgements = value)}
			/>
		{/if}
	</ConfirmDialog>
{/if}

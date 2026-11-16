<script lang="ts">
	import { onDestroy } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import { createQuery } from '@tanstack/svelte-query';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { api, ApiError } from '$lib/api/client';
	import { requestKey } from '$lib/api/upload';
	import { t, translate } from '$lib/i18n';
	import type { DraftView, ExtractionDraft, AmountPolicy, SourceRef, DraftPatch } from '$lib/types';
	import ReviewFindings from './ReviewFindings.svelte';
	import ExtractionSourceViewer from './ExtractionSourceViewer.svelte';
	import ReceiptPhotos from './ReceiptPhotos.svelte';
	import AmountPolicyEditor from './AmountPolicyEditor.svelte';
	import { hasOverlapRows, unknownOverlap } from './overlap-decisions';
	import TagEditor from './TagEditor.svelte';
	import { isSignedDecimal, isPlainDecimal } from './money';
	let { id }: { id: string } = $props();
	const tags = createQuery(
		() => ({ queryKey: queryKeys.tags(), queryFn: () => api.listTags() }),
		() => queryClient
	);
	let draft = $state<DraftView | null>(null);
	let document = $state<ExtractionDraft>({});
	let policy = $state<AmountPolicy | null>(null);
	let error = $state('');
	let busy = $state(false);
	const pendingTags = new SvelteSet<object>();
	let childPending = $derived(pendingTags.size > 0);
	function tagPending(line: object, pending: boolean) {
		if (pending) pendingTags.add(line);
		else pendingTags.delete(line);
	}
	let selectedSources = $state<SourceRef[]>([]);
	let sourceDecisions = $state<Record<string, SourceRef[]>>({});
	let overlapDecisions = $state<Record<string, unknown>[]>([]);
	let active: object | null = null;
	let finalizeKey: string | null = null;
	const fields = [
		'confidence',
		'merchant_name',
		'merchant_address',
		'receipt_number',
		'date',
		'time',
		'currency',
		'category',
		'payment_method',
		'subtotal',
		'discount_total',
		'tax_total',
		'tip',
		'total'
	] as const;
	const labelKeys: Record<string, string> = {
		confidence: 'editor.confidence',
		merchant_name: 'editor.merchant',
		merchant_address: 'editor.address',
		receipt_number: 'editor.number',
		date: 'editor.date',
		time: 'editor.time',
		currency: 'editor.currency',
		category: 'editor.category',
		payment_method: 'editor.paymentMethod',
		subtotal: 'editor.subtotal',
		discount_total: 'editor.discount',
		tax_total: 'editor.taxTotal',
		tip: 'editor.tip',
		total: 'editor.total',
		description: 'editor.description',
		quantity: 'editor.quantity',
		unit_price: 'editor.unitPrice',
		total_price: 'editor.totalPrice',
		spending_category: 'editor.spendingCategory',
		name: 'editor.taxName',
		rate: 'editor.taxRate',
		amount: 'editor.amount'
	};
	function clone<T>(value: T): T {
		return JSON.parse(JSON.stringify(value)) as T;
	}
	function adopt(value: DraftView) {
		pendingTags.clear();
		draft = value;
		document = clone(value.document);
		policy = clone(value.amount_policy);
		sourceDecisions = clone(value.field_sources ?? {});
		overlapDecisions = clone(value.overlap_decisions ?? []);
	}
	$effect(() => {
		const view = { id };
		active = view;
		draft = null;
		pendingTags.clear();
		error = '';
		busy = false;
		finalizeKey = null;
		selectedSources = [];
		void api
			.getDraft(view.id)
			.then((value) => {
				if (active === view) adopt(value);
			})
			.catch((e) => {
				if (active === view) error = String(e);
			});
		return () => {
			if (active === view) active = null;
		};
	});
	onDestroy(() => {
		active = null;
	});
	let changes = $derived.by(() => {
		const changes: DraftPatch['changes'] = {};
		if (!draft) return changes;
		for (const key of Object.keys(document) as (keyof ExtractionDraft)[]) {
			if (JSON.stringify(document[key]) !== JSON.stringify(draft.document[key]))
				Object.assign(changes, { [key]: document[key] });
		}
		if (JSON.stringify(policy) !== JSON.stringify(draft.amount_policy))
			changes.amount_policy = policy!;
		if (JSON.stringify(sourceDecisions) !== JSON.stringify(draft.field_sources ?? {}))
			changes.field_sources = sourceDecisions;
		if (JSON.stringify(overlapDecisions) !== JSON.stringify(draft.overlap_decisions ?? []))
			changes.overlap_decisions = overlapDecisions;
		return changes;
	});
	let dirty = $derived(Object.keys(changes).length > 0);
	function numeric(value: string) {
		return value.trim() === '' ? null : Number(value);
	}
	function validation(): string[] {
		const errors: string[] = [];
		for (const key of ['subtotal', 'total', 'discount_total', 'tax_total', 'tip'] as const) {
			const value = document[key];
			if (
				value != null &&
				!(key === 'subtotal' || key === 'total' ? isSignedDecimal(value) : isPlainDecimal(value))
			)
				errors.push(key);
		}
		if (
			document.confidence != null &&
			(!Number.isInteger(document.confidence) ||
				document.confidence < 0 ||
				document.confidence > 100)
		)
			errors.push('confidence');
		for (const [i, line] of (document.line_items ?? []).entries()) {
			if (line.quantity != null && (!Number.isFinite(line.quantity) || line.quantity <= 0))
				errors.push(`line_items.${i}.quantity`);
			for (const key of ['unit_price', 'total_price'] as const)
				if (line[key] != null && !isSignedDecimal(line[key]!))
					errors.push(`line_items.${i}.${key}`);
		}
		for (const [i, tax] of (document.taxes ?? []).entries()) {
			if (tax.amount != null && !isPlainDecimal(tax.amount)) errors.push(`taxes.${i}.amount`);
			if (tax.rate != null && (!Number.isFinite(tax.rate) || tax.rate < 0 || tax.rate > 1))
				errors.push(`taxes.${i}.rate`);
		}
		return errors;
	}
	async function save() {
		if (!draft || busy || childPending) return;
		const invalid = validation();
		if (invalid.length) {
			error = translate('review.invalidFields') + ': ' + invalid.join(', ');
			return;
		}
		const view = active;
		const expected = draft.revision;
		const submitted = clone(changes);
		const editSnapshot = JSON.stringify([document, policy, sourceDecisions, overlapDecisions]);
		busy = true;
		error = '';
		try {
			const result = await api.patchDraft(id, { expected_revision: expected, changes: submitted });
			if (active !== view) return;
			if (JSON.stringify([document, policy, sourceDecisions, overlapDecisions]) === editSnapshot)
				adopt(result);
			else {
				draft = result;
			}
			finalizeKey = null;
		} catch (e) {
			if (active === view)
				error =
					e instanceof ApiError && e.status === 409 ? translate('review.conflict') : String(e);
		} finally {
			if (active === view) busy = false;
		}
	}
	async function finalize() {
		if (!draft || busy || childPending) return;
		const required = [
			'confidence',
			'merchant_name',
			'date',
			'currency',
			'line_items',
			'taxes',
			'subtotal',
			'discount_total',
			'tax_total',
			'total',
			'payment_method'
		] as const;
		const missing: string[] = required.filter(
			(key) => document[key] == null || document[key] === ''
		);
		for (const [i, line] of (document.line_items ?? []).entries())
			for (const key of ['description', 'quantity', 'unit_price', 'total_price'] as const)
				if (line[key] == null || line[key] === '') missing.push(`line_items.${i}.${key}`);
		if (missing.length) {
			error = translate('review.missingFields') + ': ' + missing.join(', ');
			return;
		}
		if (dirty) {
			error = translate('review.saveFirst');
			return;
		}
		const view = active;
		busy = true;
		error = '';
		finalizeKey ??= requestKey();
		try {
			const result = await api.finalizeDraft(id, {
				expected_revision: draft.revision,
				request_key: finalizeKey
			});
			if (active === view) await goto(resolve(`/receipts/${result.id}`));
		} catch (e) {
			if (active === view)
				error =
					e instanceof ApiError && e.status === 409 ? translate('review.conflict') : String(e);
		} finally {
			if (active === view) busy = false;
		}
	}
</script>

{#if error}<p role="alert" class="my-3 text-error">{error}</p>{/if}
{#if !draft}<p>{$t('common.loading')}</p>{:else}
	<h2 class="my-4 text-xl font-semibold">{$t('review.draft')} · {draft.revision}</h2>
	{#if draft.finalized_receipt_id}<a href={resolve(`/receipts/${draft.finalized_receipt_id}`)}
			>{$t('submission.openReceipt')}</a
		>{:else}
		<fieldset disabled={busy} class="space-y-4">
			<div class="grid grid-cols-2 gap-3">
				{#each fields as field (field)}<label class="block text-sm"
						>{$t(labelKeys[field])}<input
							class="block w-full rounded border p-2"
							value={document[field] ?? ''}
							oninput={(e) =>
								Object.assign(document, {
									[field]:
										field === 'confidence'
											? numeric(e.currentTarget.value)
											: e.currentTarget.value.trim() || null
								})}
						/></label
					>{/each}
			</div>
			<h3>{$t('editor.lineItems')}</h3>
			{#each document.line_items ?? [] as line, index (line)}
				<div class="grid grid-cols-2 gap-2 rounded border p-3">
					{#each ['description', 'quantity', 'unit_price', 'total_price', 'spending_category'] as field (field)}<label
							>{$t(labelKeys[field])}<input
								class="block w-full rounded border p-2"
								value={line[field as keyof typeof line] ?? ''}
								oninput={(e) =>
									Object.assign(line, {
										[field]:
											field === 'quantity'
												? numeric(e.currentTarget.value)
												: e.currentTarget.value.trim() || null
									})}
							/></label
						>{/each}
					{#if line.tags !== null && line.tags !== undefined}<TagEditor
							disabled={busy}
							value={line.tags}
							onPendingChange={(pending) => tagPending(line, pending)}
							tagOptions={tags.data ?? []}
							id={`draft-line-${index}`}
						/>{:else}<button type="button" onclick={() => (line.tags = [])}
							>{$t('review.addTags')}</button
						>{/if}
					<button
						type="button"
						disabled={busy || childPending}
						onclick={() => {
							if (busy || childPending) return;
							document.line_items = document.line_items!.filter((_, i) => i !== index);
							overlapDecisions = overlapDecisions.map(unknownOverlap);
							sourceDecisions = Object.fromEntries(
								Object.entries(sourceDecisions).filter(([path]) => !path.startsWith('line_items'))
							);
						}}>{$t('common.remove')}</button
					>
				</div>
			{/each}
			<button
				type="button"
				class="text-primary underline"
				onclick={() =>
					(document.line_items = [
						...(document.line_items ?? []),
						{ description: null, quantity: null, unit_price: null, total_price: null, tags: null }
					])}>{$t('editor.addItem')}</button
			>
			<h3>{$t('editor.taxes')}</h3>
			{#each document.taxes ?? [] as tax, index (index)}<div class="grid grid-cols-3 gap-2">
					{#each ['name', 'rate', 'amount'] as field (field)}<label
							>{$t(labelKeys[field])}<input
								class="block w-full rounded border p-2"
								value={tax[field as keyof typeof tax] ?? ''}
								oninput={(e) =>
									Object.assign(tax, {
										[field]:
											field === 'rate'
												? numeric(e.currentTarget.value)
												: e.currentTarget.value.trim() || null
									})}
							/></label
						>{/each}<button
						type="button"
						onclick={() => {
							document.taxes = document.taxes!.filter((_, i) => i !== index);
							sourceDecisions = Object.fromEntries(
								Object.entries(sourceDecisions).filter(([path]) => !/^taxes(?:\.|\[|$)/.test(path))
							);
						}}>{$t('common.remove')}</button
					>
				</div>{/each}
			<button
				type="button"
				class="text-primary underline"
				onclick={() =>
					(document.taxes = [...(document.taxes ?? []), { name: null, rate: null, amount: null }])}
				>{$t('editor.addTax')}</button
			>
			{#if policy}<AmountPolicyEditor value={policy} onChange={(value) => (policy = value)} />{/if}
			{#each Object.entries(sourceDecisions) as [path, refs] (path)}<p>
					{path}
					<button type="button" onclick={() => (selectedSources = refs)}
						>{$t('review.showSource')}</button
					>
					<button
						type="button"
						onclick={() => (sourceDecisions = { ...sourceDecisions, [path]: [] })}
						>{$t('review.markUnknown')}</button
					>
				</p>{/each}
			{#each overlapDecisions as decision, index (index)}
				<div class="rounded-xl border border-outline-variant p-4">
					<p>{$t('review.overlapNumber', { values: { number: index + 1 } })}</p>
					{#if hasOverlapRows(decision, document.line_items?.length ?? 0, overlapDecisions)}<p>
							{$t('review.overlapRows', {
								values: {
									first: (decision.first_index as number) + 1,
									second: (decision.second_index as number) + 1
								}
							})}
						</p>{:else}<p>{$t('review.overlapUnknown')}</p>{/if}
					<label
						><input
							type="checkbox"
							disabled={!hasOverlapRows(
								decision,
								document.line_items?.length ?? 0,
								overlapDecisions
							)}
							checked={decision.verified === true}
							onchange={(e) => {
								if (hasOverlapRows(decision, document.line_items?.length ?? 0, overlapDecisions))
									overlapDecisions[index] = { ...decision, verified: e.currentTarget.checked };
							}}
						/>
						{$t('review.reviewedOverlap')}</label
					>
				</div>
			{/each}

			<div class="flex gap-4">
				<button
					type="button"
					class="rounded bg-primary p-3 text-on-primary"
					disabled={busy || childPending || !dirty}
					onclick={save}>{$t('review.saveDraft')}</button
				><button
					type="button"
					class="rounded border p-3"
					disabled={busy || childPending}
					onclick={finalize}>{$t('review.finalizeDraft')}</button
				>
			</div>
		</fieldset>
	{/if}
	<ReviewFindings
		findings={draft.findings}
		revision={draft.revision}
		editable={false}
		onSource={(sources) => (selectedSources = sources)}
	/>
	<ExtractionSourceViewer views={draft.source_views} sources={selectedSources} />
	<ReceiptPhotos imageId={id} thumbnailPath={null} alt={$t('review.original')} />
{/if}

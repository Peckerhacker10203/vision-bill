<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { onMount, onDestroy } from 'svelte';
	import { session } from '$lib/auth';
	import { api } from '$lib/api/client';
	import { captureSessionActor, isCurrentSessionActor } from '$lib/session-lifetime';
	import type { BenchmarkRun, BenchmarkStatus } from '$lib/types';
	import BenchmarkResultDetails from '$lib/ui/BenchmarkResultDetails.svelte';
	import type { IndividualResult } from '$lib/benchmark-experiments';

	const actor = captureSessionActor();
	const currentView = () => alive && isCurrentSessionActor(actor);
	let runs = $state<BenchmarkRun[]>([]),
		selectedId = $state<string | null>(null),
		result = $state<BenchmarkStatus | null>(null),
		error = $state('');
	// Admin-only page: bounce non-admins home (mirrors the /settings guard).
	$effect(() => {
		if ($session !== 'loading' && $session !== null && !$session.is_admin) {
			void goto(resolve('/'));
		}
	});
	let alive = true;
	let refreshRequest = 0,
		detailsRequest = 0;
	onDestroy(() => {
		alive = false;
	});
	let details = $state<IndividualResult[]>([]);
	let detailsOffset = $state(0),
		loadingDetails = $state(false);
	async function loadDetails(offset = 0) {
		const runId = selectedId,
			request = ++detailsRequest;
		const current = () => currentView() && selectedId === runId && detailsRequest === request;
		if (!actor || !runId || !current()) return;
		loadingDetails = true;
		try {
			const rows = await api.getBenchmarkResults(runId, offset, actor);
			if (!current()) return;
			details = rows;
			detailsOffset = offset;
		} catch (cause) {
			if (current())
				error = cause instanceof Error ? cause.message : 'Unable to load individual results.';
		} finally {
			if (current()) loadingDetails = false;
		}
	}
	const fmt = (value: number | null, digits = 2) => (value === null ? '—' : value.toFixed(digits));
	async function refresh() {
		const request = ++refreshRequest;
		const current = () => currentView() && refreshRequest === request;
		if (!actor || !current()) return;
		try {
			const available = await api.listBenchmarks(actor);
			if (!current()) return;
			runs = available;
			if (!selectedId || !runs.some((run) => run.id === selectedId)) {
				selectedId = runs[0]?.id ?? null;
				detailsRequest++;
				result = null;
				details = [];
				detailsOffset = 0;
				loadingDetails = false;
			}
			if (selectedId) {
				const next = await api.getBenchmarkStatus(selectedId, actor);
				if (!current()) return;
				result = next;
			}
			if (current()) error = '';
		} catch (cause) {
			if (current())
				error = cause instanceof Error ? cause.message : 'Unable to load benchmark results.';
		}
	}
	onMount(() => {
		void refresh();
		const timer = window.setInterval(() => void refresh(), 3000);
		return () => window.clearInterval(timer);
	});
</script>

<svelte:head><title>Benchmarks · Vision Bill</title></svelte:head>
<section class="mx-auto max-w-6xl">
	<div class="flex flex-wrap items-start justify-between gap-3">
		<div>
			<h1 class="text-2xl font-semibold">Benchmarks</h1>
			<p class="mt-1 text-sm text-on-surface-variant">Live local-model evaluation results.</p>
		</div>
		<a
			class="rounded-lg bg-error px-4 py-2.5 text-sm font-medium text-on-error"
			href={resolve('/benchmarks')}>Create benchmark</a
		>
	</div>
	{#if error}<p class="mt-4 text-sm text-error" role="alert">{error}</p>{/if}
	{#if runs.length === 0}<div
			class="mt-6 rounded-xl border border-dashed border-outline-variant p-8 text-center text-sm text-on-surface-variant"
		>
			No benchmark runs yet.
		</div>{:else}
		<label class="mt-6 block max-w-md text-sm font-medium"
			>Benchmark run<select
				class="mt-1 block w-full rounded-lg border border-outline-variant bg-surface p-2"
				value={selectedId ?? ''}
				onchange={(event) => {
					selectedId = event.currentTarget.value;
					++detailsRequest;
					loadingDetails = false;
					result = null;
					error = '';
					details = [];
					detailsOffset = 0;
					void refresh();
				}}
				>{#each runs as run (run.id)}<option value={run.id}>#{run.id} — {run.status}</option
					>{/each}</select
			></label
		>
		{#if result}
			<div class="mt-4 rounded-xl bg-surface-container p-4 text-sm">
				<p>
					Scoring version: {result.run.scoring_version}. {result.run.experiment
						? 'Extraction and classification are evaluated separately.'
						: 'Historical scoring; no stored individual outputs.'}
				</p>
				<a
					class="mt-2 inline-block underline"
					href={resolve('/benchmarks')}
					onclick={(event) => {
						event.preventDefault();
						if (!result || !currentView()) return;
						window.sessionStorage.setItem(
							'benchmark-reuse-receipts',
							result.run.receipt_ids.join(',')
						);
						void goto(resolve('/benchmarks'));
					}}>Reuse this receipt selection</a
				>
				{#if result.run.experiment}<details class="mt-2">
						<summary class="cursor-pointer">Run settings</summary>
						<pre class="mt-2 overflow-x-auto whitespace-pre-wrap text-xs">{JSON.stringify(
								result.run.experiment,
								null,
								2
							)}</pre>
					</details>{/if}
			</div>
			<div class="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
				<div class="rounded-xl bg-surface-container p-4">
					<p class="text-xs text-on-surface-variant">Completed</p>
					<p class="text-xl font-semibold">
						{result.terminal} / {result.run.receipt_ids.length * result.run.model_ids.length}
					</p>
				</div>
				<div class="rounded-xl bg-surface-container p-4">
					<p class="text-xs text-on-surface-variant">Queued</p>
					<p class="text-xl font-semibold">{result.queued}</p>
				</div>
				<div class="rounded-xl bg-surface-container p-4">
					<p class="text-xs text-on-surface-variant">Running</p>
					<p class="text-xl font-semibold">{result.running}</p>
				</div>
				<div class="rounded-xl bg-surface-container p-4">
					<p class="text-xs text-on-surface-variant">Retrying</p>
					<p class="text-xl font-semibold">{result.retrying}</p>
				</div>
				<div class="rounded-xl bg-surface-container p-4">
					<p class="text-xs text-on-surface-variant">Waiting</p>
					<p class="text-xl font-semibold">{result.waiting_for_model}</p>
				</div>
			</div>
			<div class="mt-6 overflow-x-auto rounded-xl border border-outline-variant">
				<table class="w-full text-left text-sm">
					<thead class="bg-surface-container text-xs text-on-surface-variant"
						><tr
							><th class="p-3">Model</th><th class="p-3">Correctness</th><th class="p-3"
								>Confidence</th
							><th class="p-3">Latency</th><th class="p-3">Success / failed</th><th class="p-3"
								>Council</th
							></tr
						></thead
					><tbody
						>{#each result.summaries as summary (summary.model_id)}<tr
								class="border-t border-outline-variant"
								><td class="p-3 font-medium"
									>{summary.model_id}<br /><span class="text-xs text-on-surface-variant"
										>{summary.model_digest ?? '—'}</span
									></td
								><td class="p-3"
									>{fmt(summary.average_score === null ? null : summary.average_score * 100)}%</td
								><td class="p-3">{fmt(summary.average_confidence)}%</td><td class="p-3"
									>{fmt(summary.average_latency_ms, 0)} ms</td
								><td class="p-3">{summary.succeeded} / {summary.failed}</td><td class="p-3"
									>{summary.council_findings} findings</td
								></tr
							>{/each}</tbody
					>
				</table>
			</div>
			{#if result.run.experiment}
				<div class="mt-5 flex flex-wrap items-center gap-3">
					<button
						class="rounded-lg border border-outline-variant px-3 py-2 text-sm"
						disabled={loadingDetails}
						onclick={() => loadDetails(detailsOffset)}>Load individual results</button
					>
					<button
						class="text-sm underline"
						disabled={loadingDetails || detailsOffset === 0}
						onclick={() => loadDetails(Math.max(0, detailsOffset - 50))}>Previous 50</button
					>
					<button
						class="text-sm underline"
						disabled={loadingDetails || details.length < 50}
						onclick={() => loadDetails(detailsOffset + 50)}>Next 50</button
					>
				</div>
				<BenchmarkResultDetails results={details} />
			{/if}
		{/if}
	{/if}
</section>

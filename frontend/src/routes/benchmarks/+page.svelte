<script lang="ts">
	import { t } from '$lib/i18n';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { onDestroy, onMount } from 'svelte';
	import { api } from '$lib/api/client';
	import { session } from '$lib/auth';
	import { captureSessionActor, isCurrentSessionActor } from '$lib/session-lifetime';
	import type {
		BenchmarkCouncilPolicy,
		BenchmarkCreate,
		BenchmarkReferencePreview,
		ModelInfo
	} from '$lib/types';
	import BenchmarkExperimentControls from '$lib/ui/BenchmarkExperimentControls.svelte';
	import { defaultExperiment, experimentRequest } from '$lib/benchmark-experiments';

	const actor = captureSessionActor();
	let alive = true;
	const current = () => alive && isCurrentSessionActor(actor);
	onDestroy(() => {
		alive = false;
	});

	let references = $state<BenchmarkReferencePreview[]>([]);
	let referenceSelection = $state('');
	let qualify = $state(false);
	let previewing = $state(false);
	$effect(() => {
		if (receiptIds !== referenceSelection) {
			references = [];
			qualify = false;
		}
	});
	async function previewReferences() {
		if (!actor || !current() || previewing) return;
		previewing = true;
		qualify = false;
		references = [];
		error = '';
		const selection = receiptIds;
		try {
			const ids = selection
				.split(',')
				.map((id) => id.trim())
				.filter(Boolean);
			if (!ids.length)
				throw new Error(
					'Enter explicit receipt IDs to qualify new references. Filters use existing qualified references.'
				);
			const previews = await Promise.all(ids.map((id) => api.previewBenchmarkReference(id, actor)));
			if (current() && receiptIds === selection) {
				referenceSelection = selection;
				references = previews;
			}
		} catch (cause) {
			if (current() && receiptIds === selection)
				error = cause instanceof Error ? cause.message : 'Unable to preview references.';
		} finally {
			if (current()) previewing = false;
		}
	}
	let experiment = $state(defaultExperiment());
	let capabilities = $state({ ocr_available: false, ocr_enabled: false, thinking_control: false });
	let loaded = $state(false);
	let capability: 'text' | 'vision' = $derived(
		experiment.pipeline === 'ocr' || (experiment.pipeline === 'upload' && capabilities.ocr_enabled)
			? 'text'
			: 'vision'
	);
	$effect(() => {
		if (!actor || !loaded || !current()) return;
		let active = true;
		models = [];
		selected = [];
		void api
			.listModels(capability, actor)
			.then((available) => {
				if (active && current()) {
					models = available;
					selected = available.map((model) => model.id);
				}
			})
			.catch(() => {
				if (active && current()) error = 'Unable to load local models.';
			});
		return () => {
			active = false;
		};
	});
	let models = $state<ModelInfo[]>([]),
		selected = $state<string[]>([]),
		receiptIds = $state(''),
		category = $state(''),
		maxConfidence = $state<number | undefined>(),
		limit = $state<number | undefined>(),
		timeout = $state(600),
		policy = $state<BenchmarkCouncilPolicy>('all'),
		absolute = $state<number | undefined>(),
		relative = $state<number | undefined>(),
		applyFlags = $state(false),
		confirm = $state(false),
		creating = $state(false),
		error = $state('');
	// Admin-only page: bounce non-admins home (mirrors the /settings guard).
	$effect(() => {
		if ($session !== 'loading' && $session !== null && !$session.is_admin) {
			void goto(resolve('/'));
		}
	});
	async function initialize() {
		if (!actor || !current()) return;
		receiptIds = window.sessionStorage.getItem('benchmark-reuse-receipts') ?? '';
		window.sessionStorage.removeItem('benchmark-reuse-receipts');
		const reuse = new URLSearchParams(window.location.search).get('reuse');
		if (reuse) {
			const previous = await api.getBenchmarkStatus(reuse, actor).catch(() => null);
			if (!current()) return;
			if (previous) receiptIds = previous.run.receipt_ids.join(',');
		}
		const available = await api.getBenchmarkCapabilities(actor).catch(() => null);
		if (!current()) return;
		if (available) capabilities = available;
		loaded = true;
	}
	onMount(() => {
		void initialize();
	});
	function toggle(id: string) {
		selected = selected.includes(id) ? selected.filter((model) => model !== id) : [...selected, id];
	}
	async function create() {
		if (!actor || !current() || creating) return;
		creating = true;
		error = '';
		const ids = receiptIds
			.split(',')
			.map((id) => id.trim())
			.filter(Boolean);
		const body: BenchmarkCreate = {
			model_ids: selected,
			experiment: experimentRequest(experiment),
			request_timeout_seconds: timeout,
			council_policy: policy,
			apply_council_flags: applyFlags
		};
		if (ids.length) body.receipt_ids = ids;
		if (category) body.category = category;
		if (maxConfidence !== undefined) body.max_source_confidence = maxConfidence;
		if (limit !== undefined) body.limit = limit;
		if (absolute !== undefined) body.council_absolute_threshold = absolute;
		if (relative !== undefined) body.council_relative_threshold = relative;
		try {
			if (qualify) {
				if (!references.length || receiptIds !== referenceSelection)
					throw new Error('Preview the current selection again.');
				for (const reference of references) {
					await api.qualifyBenchmarkReference(
						reference.receipt_identity,
						{
							expected_review_revision: reference.review_revision,
							expected_document_sha256: reference.document_sha256
						},
						actor
					);
					if (!current()) return;
				}
			}
			await api.createBenchmark(body, actor);
			if (current()) await goto(resolve('/benchmarks/results'));
		} catch (cause) {
			if (current()) error = cause instanceof Error ? cause.message : 'Unable to create benchmark.';
		} finally {
			if (current()) {
				creating = false;
				confirm = false;
			}
		}
	}
</script>

<svelte:head><title>Create benchmark · Vision Bill</title></svelte:head>
<section class="mx-auto max-w-3xl">
	<h1 class="text-2xl font-semibold">Create benchmark</h1>
	<p class="mt-1 text-sm text-on-surface-variant">
		Compare complete local pipelines using explicitly qualified frozen human references. Normal
		receipt data is not changed.
	</p>
	{#if error}<p class="mt-4 text-sm text-error" role="alert">{error}</p>{/if}
	<form
		class="mt-6 space-y-5"
		onsubmit={(event) => {
			event.preventDefault();
			confirm = true;
		}}
	>
		<fieldset class="rounded-xl border border-outline-variant p-4">
			<legend class="px-1 font-medium"
				>{$t(
					capability === 'text' ? 'diagnostics.formatterModels' : 'diagnostics.visionModels'
				)}</legend
			>{#each models as model (model.id)}<label class="mr-4 inline-flex items-center gap-2 py-1"
					><input
						type="checkbox"
						checked={selected.includes(model.id)}
						onchange={() => toggle(model.id)}
					/>{model.id}</label
				>{/each}
		</fieldset>
		<fieldset class="rounded-xl border border-outline-variant p-4">
			<legend class="px-1 font-medium">Verified receipt selection</legend><label
				class="block text-sm"
				>Explicit receipt IDs <input
					class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
					bind:value={receiptIds}
					placeholder="Comma-separated receipt UUIDs"
				/></label
			>
			<p class="mt-1 text-xs text-on-surface-variant">
				Explicit IDs override filters. Existing verification alone does not qualify a benchmark
				reference.
			</p>
			<button
				type="button"
				class="mt-3 rounded-lg border p-2"
				disabled={previewing}
				onclick={previewReferences}>Preview human references</button
			>
			{#if references.length}
				{#each references as reference (reference.receipt_identity)}
					<details class="mt-3 rounded border p-2">
						<summary>{reference.receipt_identity} · review {reference.review_revision}</summary>
						<a href={resolve('/receipts/[id]', { id: reference.receipt_identity })}
							>Inspect receipt and originals</a
						>
						<p>Scored observations: {reference.scored_fields.join(', ')}</p>
						<pre class="overflow-auto text-xs">{JSON.stringify(
								{
									expected: reference.expected,
									sources: reference.sources,
									findings: reference.findings
								},
								null,
								2
							)}</pre>
					</details>
				{/each}
				<label class="mt-3 block"
					><input type="checkbox" bind:checked={qualify} /> I checked these current receipts and originals
					and explicitly qualify them as benchmark references.</label
				>
			{/if}
			<div class="mt-3 grid gap-3 sm:grid-cols-3">
				<label class="text-sm"
					>Category<input
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						bind:value={category}
					/></label
				><label class="text-sm"
					>Maximum source confidence<input
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						type="number"
						min="0"
						max="100"
						bind:value={maxConfidence}
					/></label
				><label class="text-sm"
					>Maximum receipts<input
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						type="number"
						min="1"
						bind:value={limit}
					/></label
				>
			</div>
		</fieldset>
		<BenchmarkExperimentControls
			bind:settings={experiment}
			ocrAvailable={capabilities.ocr_available}
			thinkingControl={capabilities.thinking_control}
		/>
		<fieldset class="rounded-xl border border-outline-variant p-4">
			<legend class="px-1 font-medium">Execution and council</legend>
			<div class="grid gap-3 sm:grid-cols-2">
				<label class="text-sm"
					>Request timeout (seconds)<input
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						type="number"
						min="1"
						bind:value={timeout}
					/></label
				><label class="text-sm"
					>Council policy<select
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						bind:value={policy}
						><option value="all">All differences (€0.01)</option><option value="material"
							>Material (€1 and 2%)</option
						><option value="custom">Custom threshold</option></select
					></label
				><label class="text-sm"
					>Custom absolute €<input
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						type="number"
						min="0"
						step="0.01"
						bind:value={absolute}
					/></label
				><label class="text-sm"
					>Custom relative<input
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						type="number"
						min="0"
						step="0.001"
						bind:value={relative}
					/></label
				>
			</div>
			<label class="mt-3 inline-flex items-center gap-2 text-sm"
				><input type="checkbox" bind:checked={applyFlags} /> Apply council flags</label
			>
		</fieldset>
		<p class="rounded-lg bg-error-container p-3 text-sm text-on-error-container">
			<strong>Before creating:</strong> this can take a long time, consume substantial energy, and make
			the local LLM unavailable for normal receipt extraction while it runs.
		</p>
		<button
			class="rounded-lg bg-error px-4 py-2.5 text-sm font-medium text-on-error disabled:opacity-50"
			disabled={selected.length === 0 || creating}>Create benchmark</button
		>
	</form>
	{#if confirm}<div class="fixed inset-0 z-40 grid place-items-center bg-black/50 p-4">
			<div class="max-w-md rounded-xl bg-surface p-5 shadow-elevation-4">
				<h2 class="text-lg font-semibold">Create benchmark?</h2>
				<p class="mt-2 text-sm text-on-surface-variant">
					This can take a long time, consume substantial energy, and make the local LLM unavailable
					for normal receipt extraction while it runs.
				</p>
				<div class="mt-5 flex justify-end gap-2">
					<button class="rounded-lg px-4 py-2" onclick={() => (confirm = false)}>Cancel</button
					><button
						class="rounded-lg bg-error px-4 py-2 text-on-error"
						disabled={creating}
						onclick={create}>Confirm and start</button
					>
				</div>
			</div>
		</div>{/if}
</section>

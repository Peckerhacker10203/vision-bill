<script lang="ts">
	import { captureSessionActor } from '$lib/session-lifetime';
	import { api } from '$lib/api/client';
	import { t } from '$lib/i18n';
	import type { SourceView, SourceRef } from '$lib/types';
	let { views, sources = [] }: { views: SourceView[]; sources?: SourceRef[] } = $props();
	const actor = captureSessionActor();
	let failed = $state(false);
	let matching = $derived(
		views.find((v) =>
			sources.some(
				(s) =>
					s.view_id === v.view_id &&
					s.photo_index === v.photo_index &&
					s.coordinate_space === v.coordinate_space
			)
		)
	);
	let selected = $derived(matching?.view_id ?? views[0]?.view_id ?? '');
	let view = $derived(views.find((v) => v.view_id === selected) ?? views[0]);
	$effect(() => {
		void view?.view_id;
		void view?.url;
		failed = false;
	});
	let boxes = $derived(
		view
			? sources.filter(
					(s) =>
						s.view_id === view.view_id &&
						s.photo_index === view.photo_index &&
						s.coordinate_space === view.coordinate_space &&
						s.bbox &&
						s.bbox.length >= 3 &&
						s.bbox.every(
							(p) =>
								p.length === 2 &&
								Number.isFinite(p[0]) &&
								Number.isFinite(p[1]) &&
								p[0] >= 0 &&
								p[1] >= 0 &&
								p[0] <= view.width &&
								p[1] <= view.height
						)
				)
			: []
	);
</script>

<div class="my-4">
	{#if views.length > 1}<label
			>{$t('review.sourceView')}<select bind:value={selected}
				>{#each views as item (item.view_id)}<option value={item.view_id}
						>{item.photo_index + 1}: {item.coordinate_space} ({item.width} × {item.height})</option
					>{/each}</select
			></label
		>{/if}
	{#if view}
		<p class="text-xs">{view.coordinate_space} · {view.width} × {view.height} · {view.view_id}</p>
		<div class="relative w-full">
			<img
				src={api.sourceViewUrl(view.url, actor)}
				alt={$t('review.sourceView')}
				width={view.width}
				height={view.height}
				class="block h-auto w-full"
				onerror={() => (failed = true)}
			/>
			{#if !failed && boxes.length}<svg
					class="pointer-events-none absolute inset-0 h-full w-full"
					viewBox={`0 0 ${view.width} ${view.height}`}
					aria-label={$t('review.sourceBoxes')}
				>
					{#each boxes as source, index (index)}<polygon
							points={source.bbox!.map((p) => p.join(',')).join(' ')}
							fill="rgba(255,180,0,0.2)"
							stroke="#d97706"
							stroke-width="3"
						/>{/each}
				</svg>{/if}
		</div>
	{/if}
	{#if !boxes.length}<p class="text-sm">{$t('review.mappingUnknown')}</p>{/if}
	{#if failed}<p role="alert">{$t('review.sourceUnavailable')}</p>{/if}
</div>

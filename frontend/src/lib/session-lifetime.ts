import { get, writable } from 'svelte/store';

// Separate from auth/API modules so all requests can capture their origin.
export const sessionGeneration = writable(0);
export const currentSessionGeneration = () => get(sessionGeneration);
export function advanceSessionGeneration(): number {
	const next = currentSessionGeneration() + 1;
	sessionGeneration.set(next);
	return next;
}
export function requireSessionGeneration(generation: number): void {
	if (generation !== currentSessionGeneration()) {
		throw new DOMException('Session changed', 'AbortError');
	}
}

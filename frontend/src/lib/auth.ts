import { writable, type Writable } from 'svelte/store';
import { goto } from '$app/navigation';
import { resolve } from '$app/paths';
import { api, onUnauthorized } from '$lib/api/client';
import type { User } from '$lib/types';
import { queryClient } from '$lib/query/client';
import { snackbar } from '$lib/ui/snackbar.svelte';
import {
	advanceSessionGeneration,
	currentSessionGeneration,
	requireSessionGeneration
} from '$lib/session-lifetime';

/**
 * Client-side session state. `loading` until the first `/auth/me` round-trip
 * settles; `null` when signed out; the `User` when signed in. The server keeps
 * the authoritative session in the HttpOnly cookie — this store is just the
 * client's view of it for rendering and guarding.
 */
export type SessionState = User | null | 'loading';

export const session: Writable<SessionState> = writable<SessionState>('loading');

function beginSession(state: SessionState = null): number {
	const generation = advanceSessionGeneration();
	// Cancellation settles Query retryers before clearing; late query functions
	// cannot populate either the removed cache or successor observers.
	void queryClient.cancelQueries();
	queryClient.clear();
	snackbar.clear();
	sessionStorage.removeItem('benchmark-reuse-receipts');
	session.set(state);
	return generation;
}

// Browser cookies are changed by response headers before JSON is consumed.
// Serialize all auth requests (including /me) so an old Set-Cookie cannot land
// after the successor login. Generation checks alone only protect JS state.
let authTail: Promise<unknown> = Promise.resolve();
function authRequest<T>(generation: number, operation: () => Promise<T>): Promise<T> {
	const result = authTail.then(() => {
		requireSessionGeneration(generation);
		return operation();
	});
	authTail = result.catch(() => undefined);
	return result;
}

/** Populate the session from the server. Call once on app start. */
export async function initSession(): Promise<void> {
	const generation = beginSession('loading');
	try {
		const user = await authRequest(generation, () => api.me());
		requireSessionGeneration(generation);
		session.set(user);
	} catch {
		// 401 (no session) or a network error: treat as signed out.
		if (generation === currentSessionGeneration()) session.set(null);
	}
}

/** Authenticate with username/password and record the returned user. */
export async function signIn(username: string, password: string): Promise<User> {
	const generation = beginSession();
	const user = await authRequest(generation, () => api.login(username, password));
	requireSessionGeneration(generation);
	session.set(user);
	return user;
}

/** Create an account and sign in with it. */
export async function signUp(username: string, password: string): Promise<User> {
	const generation = beginSession();
	const user = await authRequest(generation, () => api.register(username, password));
	requireSessionGeneration(generation);
	session.set(user);
	return user;
}

/** Clear the session (server cookie via the endpoint + local state).

A failing logout endpoint (e.g. offline) must not block sign-out: the local
session is cleared regardless and the caller redirects to the login page. */
export async function signOut(): Promise<void> {
	const generation = beginSession();
	try {
		await authRequest(generation, () => api.logout());
	} catch {
		// Ignore: clearing the local session is what actually signs the user out.
	}
}

// A 401 from a protected endpoint means the session is gone: forget the user
// locally and return to the login page. Registered here (not in the layout) so
// the handler is active for every request, including background refetches.
onUnauthorized(() => {
	beginSession();
	void goto(resolve('/login'));
});

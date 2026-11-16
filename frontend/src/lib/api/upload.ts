import { sha256 } from '@noble/hashes/sha256';
import { bytesToHex, utf8ToBytes } from '@noble/hashes/utils';
import type { UploadOptions } from '$lib/types';

/** Only opaque request identity and explicit options enter browser storage. */
export interface PendingUpload {
	key: string;
	identity: string;
	options: UploadOptions;
}
const storageKey = (owner: string) => `vb-upload:${owner}`;

export function requestKey(): string {
	// getRandomValues is available on local HTTP; subtle/randomUUID require HTTPS.
	return bytesToHex(crypto.getRandomValues(new Uint8Array(24)));
}

export function readPendingUpload(owner: string): PendingUpload | null {
	try {
		const value = JSON.parse(localStorage.getItem(storageKey(owner)) ?? 'null');
		if (!value || typeof value.key !== 'string' || !/^[a-f0-9]{64}$/.test(value.identity))
			return null;
		const options = value.options;
		if (!options || typeof options !== 'object' || Array.isArray(options)) return null;
		if (Object.keys(options).some((key) => key !== 'bypass_review' && key !== 'model_id'))
			return null;
		if ('bypass_review' in options && typeof options.bypass_review !== 'boolean') return null;
		if ('model_id' in options && typeof options.model_id !== 'string') return null;
		return { key: value.key, identity: value.identity, options };
	} catch {
		return null;
	}
}

function fileBytes(file: File): Promise<ArrayBuffer> {
	if (file.arrayBuffer) return file.arrayBuffer();
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(reader.result as ArrayBuffer);
		reader.onerror = () => reject(new Error('Could not read selected photo'));
		reader.readAsArrayBuffer(file);
	});
}

/** Call only after an explicit upload click, never on recovery/mount. */
export async function prepareUpload(
	owner: string,
	files: File[],
	explicitOptions?: UploadOptions
): Promise<PendingUpload> {
	const pending = readPendingUpload(owner);
	// An untouched control (including a changed server default) cannot alter a replay.
	const selected = explicitOptions ?? pending?.options ?? {};
	const options: UploadOptions = {};
	if (selected.model_id !== undefined) options.model_id = selected.model_id;
	if (selected.bypass_review !== undefined) options.bypass_review = selected.bypass_review;
	const hashes: string[] = [];
	for (const file of files) hashes.push(bytesToHex(sha256(new Uint8Array(await fileBytes(file)))));
	const identity = bytesToHex(sha256(utf8ToBytes(JSON.stringify({ owner, hashes, options }))));
	if (pending?.identity === identity) return pending;
	const next = { key: requestKey(), identity, options };
	// Persist before POST. If storage is blocked, fail without creating an unrecoverable job.
	localStorage.setItem(storageKey(owner), JSON.stringify(next));
	return next;
}

export function clearPendingUpload(owner: string, key: string): void {
	if (readPendingUpload(owner)?.key === key) localStorage.removeItem(storageKey(owner));
}

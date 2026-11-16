import { afterEach, expect, it, vi } from 'vitest';
import { api } from './client';
import { prepareUpload, readPendingUpload, clearPendingUpload } from './upload';
afterEach(() => {
	localStorage.clear();
	vi.unstubAllGlobals();
});
const files = () => [new File(['private one'], 'same.png'), new File(['private two'], 'same.png')];
it('reuses key and first explicit options after network and truncated-success ambiguity, including reload', async () => {
	const input = files();
	const first = await prepareUpload('owner', input, { bypass_review: false });
	vi.stubGlobal(
		'fetch',
		vi
			.fn()
			.mockRejectedValueOnce(new Error('offline'))
			.mockResolvedValueOnce({
				ok: true,
				status: 202,
				json: async () => {
					throw new SyntaxError('truncated');
				}
			})
	);
	await expect(api.uploadImages(input, first.options, first.key)).rejects.toBeTruthy();
	const retry = await prepareUpload('owner', input);
	expect(retry).toEqual(first);
	await expect(api.uploadImages(input, retry.options, retry.key)).rejects.toMatchObject({
		status: 202
	});
	expect(readPendingUpload('owner')).toEqual(first);
	expect((await prepareUpload('owner', files())).key).toBe(first.key);
	expect(fetch).toHaveBeenCalledTimes(2);
	expect(localStorage.getItem('vb-upload:owner')).not.toMatch(/private|same.png|blob:/);
});
it('changes key for changed content, order, explicit options/presence and owner', async () => {
	const input = files();
	const first = await prepareUpload('a', input, {});
	const reordered = await prepareUpload('a', [...input].reverse(), {});
	expect(reordered.key).not.toBe(first.key);
	const explicit = await prepareUpload('a', [...input].reverse(), { bypass_review: false });
	expect(explicit.key).not.toBe(reordered.key);
	expect((await prepareUpload('b', [...input].reverse(), { bypass_review: false })).key).not.toBe(
		explicit.key
	);
	const changed = await prepareUpload('a', [new File(['different'], 'same.png'), input[0]], {
		bypass_review: false
	});
	expect(changed.key).not.toBe(explicit.key);
	clearPendingUpload('a', changed.key);
	expect(readPendingUpload('a')).toBeNull();
});
it('hashes actual bytes and generates keys on local HTTP without subtle or randomUUID', async () => {
	vi.stubGlobal('crypto', {
		getRandomValues: (bytes: Uint8Array) => {
			for (let i = 0; i < bytes.length; i++) bytes[i] = i;
			return bytes;
		}
	});
	const one = await prepareUpload('a', files(), {});
	expect(one.key.length).toBeGreaterThanOrEqual(32);
	expect(one.identity).toMatch(/^[0-9a-f]{64}$/);
	expect((await prepareUpload('a', files(), {})).key).toBe(one.key);
});

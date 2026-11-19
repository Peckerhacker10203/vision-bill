import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { api } from '$lib/api/client';
import { session } from '$lib/auth';
import { queryClient } from '$lib/query/client';
import Settings from './+page.svelte';

vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));

const admin = { id: 'admin', username: 'admin', is_admin: true, can_see_all: true };
const directVision = {
	bypass_review_default: false,
	registration_open: true,
	ocr_enabled: false,
	formatter_max_output_tokens: 8192
};

beforeEach(() => {
	session.set(admin);
	vi.spyOn(api, 'getSettings').mockResolvedValue({
		llm: { provider: 'ollama', host: 'local', model_name: 'configured', temperature: 0 },
		allow_registration: true,
		sources: {},
		restart_required: false
	});
	vi.spyOn(api, 'getUiConfig').mockResolvedValue(directVision);
	vi.spyOn(api, 'listModels').mockResolvedValue([]);
});

afterEach(() => {
	cleanup();
	queryClient.clear();
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
});

function status(state: string, reason: string | null = null, httpStatus = 200) {
	return Response.json({ state, reason }, { status: httpStatus });
}

function handoffConfirmation() {
	return screen.findByRole('checkbox', {
		name: 'I confirm the LLM endpoint is dedicated to Vision Bill and idle, with eager model loading turned off.'
	});
}

it('does not load Dedicated OCR setup for a non-admin who visits Settings', async () => {
	session.set({ ...admin, is_admin: false, can_see_all: false });
	const fetchStatus = vi.fn();
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	await screen.findByRole('heading', { name: 'Settings' });

	expect(screen.queryByRole('region', { name: 'Dedicated OCR' })).toBeNull();
	expect(fetchStatus).not.toHaveBeenCalled();
});

it('shows that a GPU profile is required and offers no activation', async () => {
	const fetchStatus = vi.fn().mockResolvedValue(status('unavailable', 'gpu_profile_required'));
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	await screen.findByText('Select a CUDA or ROCm profile to use Dedicated OCR.');

	expect(screen.queryByRole('button', { name: 'Enable Dedicated OCR' })).toBeNull();
	expect(fetchStatus).toHaveBeenCalledOnce();
	const [url, options] = fetchStatus.mock.calls[0];
	expect(url).toBe('/api/v1/system/dedicated-ocr');
	expect(options.method).toBeUndefined();
	expect(options.headers.get('X-Vision-Bill-Expected-User')).toBe('admin');
});

it('submits one explicit activation and polls until the formatter switches to text models', async () => {
	let finishRequest!: (response: Response) => void;
	const post = new Promise<Response>((resolve) => (finishRequest = resolve));
	const fetchStatus = vi.fn(async (_url: string, options: RequestInit) =>
		options.method === 'POST' ? post : status('disabled')
	);
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	const enable = await screen.findByRole('button', { name: 'Enable Dedicated OCR' });
	await waitFor(() => expect(api.listModels).toHaveBeenCalledWith('vision'));
	expect((enable as HTMLButtonElement).disabled).toBe(true);
	await fireEvent.click(enable);
	expect(fetchStatus).toHaveBeenCalledOnce();
	await fireEvent.click(await handoffConfirmation());

	await Promise.all([fireEvent.click(enable), fireEvent.click(enable)]);
	const posts = fetchStatus.mock.calls.filter(([, options]) => options.method === 'POST');
	expect(posts).toHaveLength(1);
	expect(posts[0][0]).toBe('/api/v1/system/dedicated-ocr');
	expect(new Headers(posts[0][1].headers).get('X-Vision-Bill-Expected-User')).toBe('admin');
	expect(posts[0][1].body).toBe('{"confirm_handoff":true}');
	expect((enable as HTMLButtonElement).disabled).toBe(true);

	finishRequest(status('requested', null, 202));
	await screen.findByText('Dedicated OCR setup is queued.');
	vi.mocked(api.getUiConfig).mockResolvedValue({ ...directVision, ocr_enabled: true });
	fetchStatus.mockResolvedValueOnce(status('installing')).mockResolvedValueOnce(status('active'));
	await screen.findByText(
		'Downloading and installing Dedicated OCR models…',
		{},
		{ timeout: 3_000 }
	);
	expect(api.listModels).not.toHaveBeenCalledWith('text');
	await screen.findByText('Dedicated OCR is active.', {}, { timeout: 3_000 });
	await waitFor(() => expect(api.listModels).toHaveBeenCalledWith('text'));
	const requestCount = fetchStatus.mock.calls.length;
	await new Promise((resolve) => setTimeout(resolve, 2_100));
	expect(fetchStatus).toHaveBeenCalledTimes(requestCount);
}, 10_000);

it('offers an explicit retry after a failed download', async () => {
	const fetchStatus = vi
		.fn()
		.mockResolvedValueOnce(status('failed', 'ocr_download_failed'))
		.mockResolvedValue(status('requested', null, 202));
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	await screen.findByText('Dedicated OCR model download failed.');
	const retry = await screen.findByRole('button', { name: 'Retry setup' });
	expect((retry as HTMLButtonElement).disabled).toBe(true);
	await fireEvent.click(retry);
	expect(fetchStatus).toHaveBeenCalledOnce();
	await fireEvent.click(await handoffConfirmation());
	await fireEvent.click(retry);
	await screen.findByText('Dedicated OCR setup is queued.');
	expect(fetchStatus.mock.calls[1][1].method).toBe('POST');
});

it('refreshes status when activation conflicts with an operator change', async () => {
	const fetchStatus = vi
		.fn()
		.mockResolvedValueOnce(status('disabled'))
		.mockResolvedValueOnce(Response.json({ detail: 'llm_lifecycle_required' }, { status: 409 }))
		.mockResolvedValueOnce(status('action_required', 'llm_lifecycle_required'))
		.mockResolvedValueOnce(status('requested', null, 202));
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	await fireEvent.click(await handoffConfirmation());
	await fireEvent.click(await screen.findByRole('button', { name: 'Enable Dedicated OCR' }));
	await screen.findByText('Configure a supported LLM lifecycle before setting up Dedicated OCR.');
	const retry = await screen.findByRole('button', { name: 'Retry setup' });
	await fireEvent.click(retry);
	expect(fetchStatus).toHaveBeenCalledTimes(3);
	await fireEvent.click(await handoffConfirmation());
	await fireEvent.click(retry);
	await screen.findByText('Dedicated OCR setup is queued.');
	expect(fetchStatus.mock.calls[3][1].method).toBe('POST');
});

it('checks an unavailable installed runtime without offering another setup', async () => {
	const fetchStatus = vi
		.fn()
		.mockResolvedValueOnce(status('action_required', 'ocr_runtime_unavailable'))
		.mockResolvedValueOnce(status('active'));
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	await screen.findByText(
		'Dedicated OCR is installed. Restore its service before processing receipts.'
	);
	expect(screen.queryByRole('button', { name: 'Retry setup' })).toBeNull();
	await fireEvent.click(screen.getByRole('button', { name: 'Check status again' }));
	await screen.findByText('Dedicated OCR is active.');
	expect(fetchStatus).toHaveBeenCalledTimes(2);
	expect(fetchStatus.mock.calls[1][1].method).toBeUndefined();
});

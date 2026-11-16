import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, expect, it, vi } from 'vitest';
import { api } from '$lib/api/client';
import { queryClient } from '$lib/query/client';
import { session } from '$lib/auth';
import Settings from './+page.svelte';
vi.mock('$app/paths', () => ({ resolve: (p: string) => p }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));
afterEach(() => {
	cleanup();
	queryClient.clear();
	vi.restoreAllMocks();
});
it.each([true, false])(
	'discovers appropriate models with OCR enabled=%s, preserving typed selection and distinct limits',
	async (ocr) => {
		session.set({ id: 'admin', username: 'admin', is_admin: true, can_see_all: true });
		vi.spyOn(api, 'getSettings').mockResolvedValue({
			llm: { provider: 'ollama', host: 'local', model_name: 'configured', temperature: 0 },
			allow_registration: true,
			sources: {},
			restart_required: false
		});
		vi.spyOn(api, 'getUiConfig').mockResolvedValue({
			bypass_review_default: false,
			registration_open: true,
			ocr_enabled: ocr,
			formatter_max_output_tokens: 8192
		});
		vi.spyOn(api, 'listModels').mockResolvedValue([
			{ id: 'formatter', context_window_tokens: 32768 }
		]);
		render(Settings);
		await waitFor(() => expect(api.listModels).toHaveBeenCalledWith(ocr ? 'text' : 'vision'));
		const model = screen.getByLabelText(/Default model/);
		await fireEvent.input(model, { target: { value: 'custom-model' } });
		await queryClient.invalidateQueries({ queryKey: ['models'] });
		expect((model as HTMLInputElement).value).toBe('custom-model');
		expect(model.getAttribute('list')).toBe('formatter-models');
		expect(screen.getByText(/8192/)).toBeTruthy();
		expect(document.querySelector('option[value="formatter"]')?.textContent).toContain('32768');
	}
);

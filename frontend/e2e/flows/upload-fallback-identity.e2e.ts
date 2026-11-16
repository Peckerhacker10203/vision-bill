import { expect, test } from '@playwright/test';
import { registerUser } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { testEnvironment } from '../helpers/environment';
import { setStubMode, stubGuard } from '../helpers/stub';

test.describe('explicit fallback permission recovery', () => {
	stubGuard();
	test('ambiguous acceptance replays exact permission; changing one control preserves the other', async ({
		page,
		context
	}) => {
		test.skip(
			testEnvironment('VB_E2E_POLICY') !== 'fallback',
			'Requires the explicitly permitted fallback stack'
		);
		await registerUser(context, 'fallback-identity');
		const fixture = await loadFixture('a');
		await setStubMode('down');
		const attempts: { key: string | undefined; options: string; id: string }[] = [];
		await page.route('**/api/v1/images?*', async (route) => {
			if (route.request().method() !== 'POST') return route.continue();
			const response = await route.fetch();
			expect(response.status()).toBe(202);
			const accepted = await response.json();
			attempts.push({
				key: route.request().headers()['idempotency-key'],
				options: new URL(route.request().url()).search,
				id: accepted.id
			});
			// Acceptance happened on the real server; simulate losing its identifying response.
			if (attempts.length <= 2) await route.fulfill({ status: 202, json: {} });
			else await route.fulfill({ response });
		});
		await page.goto('/upload');
		const permission = page.getByLabel('Allow vision analysis if OCR fails', { exact: true });
		await expect(permission).not.toBeChecked();
		await permission.check();
		await page.getByLabel(/Skip review/).check();
		const select = async () =>
			page
				.locator('input[type="file"]')
				.first()
				.setInputFiles({ name: 'a.png', mimeType: fixture.mimeType, buffer: fixture.bytes });
		await select();
		await page.getByRole('button', { name: 'Upload', exact: true }).click();
		await expect(
			page.getByText('Response did not include a submission ID', { exact: true })
		).toBeVisible();
		await page.reload();
		await expect(permission).toBeChecked();
		await expect(page.getByLabel(/Skip review/)).toBeChecked();
		await select();
		await page.getByRole('button', { name: 'Upload', exact: true }).click();
		await expect.poll(() => attempts.length).toBe(2);
		expect(attempts[1]).toEqual(attempts[0]);
		await page.getByLabel(/Skip review/).uncheck();
		await page.getByRole('button', { name: 'Upload', exact: true }).click();
		await expect(page).toHaveURL(/\/upload\?job=/);
		expect(attempts).toHaveLength(3);
		expect(attempts[2].key).not.toBe(attempts[0].key);
		expect(attempts[2].id).not.toBe(attempts[0].id);
		const options = new URLSearchParams(attempts[2].options);
		expect(options.get('allow_ocr_fallback')).toBe('true');
		expect(options.get('bypass_review')).toBe('false');
		for (const id of new Set(attempts.map((a) => a.id))) {
			const job = await (await context.request.get(`/api/v1/images/${id}`)).json();
			expect(job.status).toBe('pending');
			expect(job.attempt_count).toBe(0);
			expect((await context.request.delete(`/api/v1/images/${id}`)).status()).toBe(200);
		}
	});
});

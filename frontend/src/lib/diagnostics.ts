import en from '$lib/i18n/en.json';
import type { translate } from '$lib/i18n';

export function diagnosticLabel(
	t: (...args: Parameters<typeof translate>) => ReturnType<typeof translate>,
	group: 'checks' | 'reasons' | 'severity' | 'comparison',
	code: string
): string {
	if (Object.hasOwn(en.diagnostics[group], code)) return t(`diagnostics.${group}.${code}`);
	return t('diagnostics.unknown', { values: { label: code.replace(/[_-]+/g, ' ') } });
}

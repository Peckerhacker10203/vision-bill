/** Mutable row positions must never stand in for immutable extraction evidence. */
export function hasOverlapRows(decision: Record<string, unknown>, count: number): boolean {
	const first = decision.first_index,
		second = decision.second_index;
	return (
		decision.association_unknown !== true &&
		typeof first === 'number' &&
		Number.isInteger(first) &&
		typeof second === 'number' &&
		Number.isInteger(second) &&
		first >= 0 &&
		second >= 0 &&
		first < count &&
		second < count &&
		first !== second
	);
}
export function unknownOverlap(decision: Record<string, unknown>): Record<string, unknown> {
	const result = { ...decision, verified: false, association_unknown: true };
	for (const key of ['first_index', 'second_index', 'removed_index'])
		delete (result as Record<string, unknown>)[key];
	return result;
}
export function currentOverlaps(
	decisions: Record<string, unknown>[],
	count: number
): Record<string, unknown>[] {
	return decisions.map((decision) =>
		hasOverlapRows(decision, count) ? decision : unknownOverlap(decision)
	);
}

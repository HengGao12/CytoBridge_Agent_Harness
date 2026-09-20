/**
 * Lightweight `classNames` helper.
 *
 * Lives in its own module so the UI primitives barrel (`./index.jsx`) only
 * exports React components — required by `react-refresh/only-export-components`.
 */
export function cn(...classes) {
    return classes.filter(Boolean).join(' ');
}

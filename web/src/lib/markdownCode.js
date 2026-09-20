/**
 * Detect whether a react-markdown `code` node is inline.
 *
 * react-markdown v10 dropped the `inline` prop that older code relied on, so
 * the `code` component is invoked for both inline spans and fenced blocks.
 * Fenced blocks carry a `language-*` className and/or a trailing newline in
 * their text; inline spans never contain a newline. That distinction is
 * sufficient and avoids reaching into the mdast `node` shape.
 */
export function isInlineCode(className, children) {
  if (className && /\blanguage-/.test(className)) return false;
  const text = Array.isArray(children)
    ? children.join('')
    : String(children ?? '');
  return !text.includes('\n');
}

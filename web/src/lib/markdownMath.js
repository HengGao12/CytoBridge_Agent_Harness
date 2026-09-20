const FENCE_RE = /^ {0,3}(`{3,}|~{3,})/;

const splitFencedBlocks = (text) => {
  const lines = String(text || '').split(/(\n)/);
  const segments = [];
  let buffer = '';
  let inFence = false;
  let fenceMarker = '';

  const flush = () => {
    if (!buffer) return;
    segments.push({ fenced: inFence, text: buffer });
    buffer = '';
  };

  for (let idx = 0; idx < lines.length; idx += 2) {
    const line = lines[idx] || '';
    const newline = lines[idx + 1] || '';
    const match = line.match(FENCE_RE);

    if (match && !inFence) {
      flush();
      inFence = true;
      fenceMarker = match[1][0];
      buffer += line + newline;
      continue;
    }

    if (match && inFence && match[1][0] === fenceMarker) {
      buffer += line + newline;
      flush();
      inFence = false;
      fenceMarker = '';
      continue;
    }

    buffer += line + newline;
  }

  flush();
  return segments;
};

const splitInlineCode = (text) => {
  const parts = [];
  let cursor = 0;

  while (cursor < text.length) {
    const tickStart = text.indexOf('`', cursor);
    if (tickStart < 0) {
      parts.push({ code: false, text: text.slice(cursor) });
      break;
    }

    if (tickStart > cursor) {
      parts.push({ code: false, text: text.slice(cursor, tickStart) });
    }

    let tickEnd = tickStart;
    while (tickEnd < text.length && text[tickEnd] === '`') tickEnd += 1;
    const fence = text.slice(tickStart, tickEnd);
    const closing = text.indexOf(fence, tickEnd);
    if (closing < 0) {
      parts.push({ code: false, text: text.slice(tickStart) });
      break;
    }

    parts.push({ code: true, text: text.slice(tickStart, closing + fence.length) });
    cursor = closing + fence.length;
  }

  return parts;
};

const normalizeDisplayEnvironment = (env, body) => {
  const trimmed = String(body || '').trim();
  const envName = String(env || '').replace(/\*$/, '');
  if (envName === 'align') {
    return `\n$$\n\\begin{aligned}\n${trimmed}\n\\end{aligned}\n$$\n`;
  }
  return `\n$$\n${trimmed}\n$$\n`;
};

const splitDisplayMath = (text) => {
  const parts = [];
  let cursor = 0;

  while (cursor < text.length) {
    const start = text.indexOf('$$', cursor);
    if (start < 0) {
      parts.push({ math: false, text: text.slice(cursor) });
      break;
    }

    if (start > cursor) {
      parts.push({ math: false, text: text.slice(cursor, start) });
    }

    const end = text.indexOf('$$', start + 2);
    if (end < 0) {
      parts.push({ math: false, text: text.slice(start) });
      break;
    }

    parts.push({ math: true, text: text.slice(start, end + 2) });
    cursor = end + 2;
  }

  return parts;
};

const normalizeDisplayDelimiters = (text) => String(text || '')
  .replace(/\\\[((?:.|\n)*?)\\\]/g, (_, body) => `\n$$\n${String(body || '').trim()}\n$$\n`)
  .replace(/(^|[^$])\$\$([^\n$][\s\S]*?[^\n$])\$\$(?!\$)/g, (_, prefix, body) => (
    `${prefix}\n$$\n${String(body || '').trim()}\n$$\n`
  ));

const normalizeMathOutsideDisplay = (text) => String(text || '')
  .replace(
    /\\begin\{(equation\*?|align\*?)\}((?:.|\n)*?)\\end\{\1\}/g,
    (_, env, body) => normalizeDisplayEnvironment(env, body),
  )
  .replace(
    /\\begin\{(aligned|gathered|cases|matrix|pmatrix|bmatrix|array)\}((?:.|\n)*?)\\end\{\1\}/g,
    (_, env, body) => `\n$$\n\\begin{${env}}\n${String(body || '').trim()}\n\\end{${env}}\n$$\n`,
  )
  .replace(/\\\(([^\n]*?)\\\)/g, (_, body) => `$${String(body || '').trim()}$`);

const normalizeMathInText = (text) => splitDisplayMath(normalizeDisplayDelimiters(text))
  .map((part) => (part.math ? part.text : normalizeMathOutsideDisplay(part.text)))
  .join('');

const normalizeNonFencedSegment = (text) => splitInlineCode(text)
  .map((part) => (part.code ? part.text : normalizeMathInText(part.text)))
  .join('');

const GFM_TABLE_SEPARATOR_ROW_RE = /\|\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|/;

const normalizeInlinePipeTableLine = (line) => {
  if (!GFM_TABLE_SEPARATOR_ROW_RE.test(line)) {
    return line;
  }

  let normalized = String(line || '').replace(/\|\s+(?=\|)/g, '|\n');
  const tableStart = normalized.indexOf('|');
  if (tableStart > 0 && normalized.slice(0, tableStart).trim()) {
    normalized = `${normalized.slice(0, tableStart).trimEnd()}\n\n${normalized.slice(tableStart)}`;
  }
  return normalized;
};

const normalizeInlinePipeTablesInText = (text) => String(text || '')
  .split('\n')
  .map(normalizeInlinePipeTableLine)
  .join('\n');

const normalizeTablesNonFencedSegment = (text) => splitInlineCode(text)
  .map((part) => (part.code ? part.text : normalizeInlinePipeTablesInText(part.text)))
  .join('');

export const normalizeMarkdownTables = (markdown) => splitFencedBlocks(markdown)
  .map((segment) => (segment.fenced ? segment.text : normalizeTablesNonFencedSegment(segment.text)))
  .join('');

export const normalizeMathMarkdown = (markdown) => splitFencedBlocks(markdown)
  .map((segment) => (segment.fenced ? segment.text : normalizeNonFencedSegment(segment.text)))
  .join('');

export const normalizeMarkdownForRendering = (markdown) => normalizeMathMarkdown(
  normalizeMarkdownTables(markdown),
);

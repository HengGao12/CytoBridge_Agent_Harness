import assert from 'node:assert/strict';
import {
  normalizeMarkdownForRendering,
  normalizeMarkdownTables,
  normalizeMathMarkdown,
} from './markdownMath.js';

{
  const output = normalizeMathMarkdown('Inline \\(x^2 + y^2\\) works.');
  assert.equal(output, 'Inline $x^2 + y^2$ works.');
}

{
  const output = normalizeMathMarkdown('Display:\n\\[\\int_0^1 x\\,dx\\]');
  assert.match(output, /\$\$\n\\int_0\^1 x\\,dx\n\$\$/);
}

{
  const output = normalizeMathMarkdown('Display $$x_t = v_\\theta(x,t)$$ after text.');
  assert.match(output, /Display \n\$\$\nx_t = v_\\theta\(x,t\)\n\$\$\n after text\./);
}

{
  const input = 'Do not touch `\\(not math\\)` or:\n```md\n\\[not math\\]\n```';
  assert.equal(normalizeMathMarkdown(input), input);
}

{
  const output = normalizeMathMarkdown('\\begin{align}a &= b \\\\ c &= d\\end{align}');
  assert.match(output, /\\begin\{aligned\}/);
  assert.match(output, /\\end\{aligned\}/);
}

{
  const output = normalizeMathMarkdown('\\begin{cases}x & x > 0\\\\0 & x \\le 0\\end{cases}');
  assert.match(output, /\$\$\n\\begin\{cases\}/);
  assert.match(output, /\\end\{cases\}\n\$\$/);
}

{
  const input = '| 领域 | 薛定谔桥的解释 | |---|---| | 统计物理 | 端点约束下最可能的粒子系统演化 | | 信息论 | 最小相对熵修正 |';
  const output = normalizeMarkdownTables(input);
  assert.equal(
    output,
    '| 领域 | 薛定谔桥的解释 |\n|---|---|\n| 统计物理 | 端点约束下最可能的粒子系统演化 |\n| 信息论 | 最小相对熵修正 |',
  );
}

{
  const input = '表格如下：| A | B | |---|---| | 1 | 2 |';
  const output = normalizeMarkdownForRendering(input);
  assert.equal(output, '表格如下：\n\n| A | B |\n|---|---|\n| 1 | 2 |');
}

{
  const input = 'Do not touch:\n```md\n| A | B | |---|---| | 1 | 2 |\n```';
  assert.equal(normalizeMarkdownTables(input), input);
}

// Pair only character matches shared by every longest common subsequence.
// Ambiguous repeated substrings remain unpaired rather than inventing correspondence.
export function alignTokens(a, b) {
  const x = a.prompt,
    y = b.prompt,
    n = x.length,
    m = y.length;
  if (n * m > 8_000_000) return a.tokens.map(() => null);
  const before = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  const after = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++)
      before[i + 1][j + 1] =
        x[i] === y[j]
          ? before[i][j] + 1
          : Math.max(before[i][j + 1], before[i + 1][j]);
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      after[i][j] =
        x[i] === y[j]
          ? after[i + 1][j + 1] + 1
          : Math.max(after[i + 1][j], after[i][j + 1]);
  const length = after[0][0],
    candidates = Array.from({ length: n }, () => []),
    inverse = Array.from({ length: m }, () => []);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++)
      if (x[i] === y[j] && before[i][j] + 1 + after[i + 1][j + 1] === length) {
        candidates[i].push(j);
        inverse[j].push(i);
      }
  const required = candidates.map((_, i) => {
    let best = 0;
    for (let j = 0; j <= m; j++)
      best = Math.max(best, before[i][j] + after[i + 1][j]);
    return best < length;
  });
  return a.tokens.map((t) => {
    if (t.boundary) return b.tokens[0]?.boundary ? 0 : null;
    const [start, end] = t.offset;
    if (start === end) return null;
    const matches = candidates.slice(start, end);
    if (matches.some((c, k) => c.length !== 1 || !required[start + k]))
      return null;
    const target = matches.map((c) => c[0]);
    if (target.some((j, k) => inverse[j].length !== 1 || j !== target[0] + k))
      return null;
    const index = b.tokens.findIndex(
      (u) =>
        !u.boundary &&
        u.id === t.id &&
        u.offset[0] === target[0] &&
        u.offset[1] === target.at(-1) + 1 &&
        y.slice(...u.offset) === x.slice(start, end),
    );
    return index < 0 ? null : index;
  });
}

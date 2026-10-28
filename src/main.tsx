import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { Run, Head, Gallery } from "./types";
import { alignTokens } from "./alignment.mjs";
import "./style.css";
import { saveCard } from "./card";
const show = (s: string) => s.replaceAll(" ", "·").replaceAll("\n", "↵");
const pct = (p: number) => `${(100 * p).toFixed(p < 0.01 ? 3 : 1)}%`;
const empty: Head = { attention: [], q: [], k: [] };
const session = sessionStorage.getItem("xray-session") || crypto.randomUUID();
sessionStorage.setItem("xray-session", session);
function Matrix({
  head,
  query,
  onQuery,
  label,
}: {
  head: Head;
  query: number;
  onQuery: (q: number) => void;
  label: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [tip, setTip] = useState("");
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const n = head.attention.length,
      size = 420,
      dpr = Math.min(devicePixelRatio, 2);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    const c = canvas.getContext("2d")!;
    c.scale(dpr, dpr);
    c.clearRect(0, 0, size, size);
    const cell = 360 / Math.max(1, n);
    head.attention.forEach((row, q) =>
      row.forEach((v, k) => {
        c.fillStyle =
          k > q
            ? "#172126"
            : `rgb(${32 + 98 * Math.sqrt(v)},${49 + 168 * Math.sqrt(v)},${52 + 151 * Math.sqrt(v)})`;
        c.fillRect(38 + k * cell, 20 + q * cell, cell - 0.7, cell - 0.7);
      }),
    );
    c.strokeStyle = "#e6b87b";
    c.lineWidth = 1;
    c.strokeRect(37, 19 + query * cell, 362, cell + 2);
    c.font = "9px monospace";
    c.fillStyle = "#95a7ab";
    c.textAlign = "center";
    for (let i = 0; i < n; i++)
      if (n < 28 || i % Math.ceil(n / 24) === 0) {
        c.fillText(String(i), 38 + (i + 0.5) * cell, 398);
        c.fillText(String(i), 23, 20 + (i + 0.6) * cell);
      }
  }, [head, query]);
  const cell = (e: React.MouseEvent) => {
    const r = e.currentTarget.getBoundingClientRect(),
      n = head.attention.length;
    return [
      Math.floor(((((e.clientX - r.left) / r.width) * 420 - 38) / 360) * n),
      Math.floor(((((e.clientY - r.top) / r.height) * 420 - 20) / 360) * n),
    ];
  };
  return (
    <div className="matrix">
      <canvas
        ref={ref}
        aria-label={label}
        role="img"
        onMouseMove={(e) => {
          const [k, q] = cell(e);
          setTip(
            head.attention[q]?.[k] !== undefined
              ? `${q} → ${k} · ${k > q ? "causally masked" : pct(head.attention[q][k])}`
              : "",
          );
        }}
        onClick={(e) => {
          const [, q] = cell(e);
          if (q >= 0 && q < head.attention.length) onQuery(q);
        }}
      />
      <div className="plot-foot">{tip || "Query ↓ · Key → · select a row"}</div>
    </div>
  );
}
function Difference({
  a,
  b,
  alignment,
}: {
  a: Head;
  b: Head;
  alignment: (number | null)[];
}) {
  const paired = alignment.flatMap((j, i) => (j === null ? [] : [{ i, j }]));
  return (
    <details className="difference">
      <summary>Matched-token attention difference · B − A</summary>
      <div
        className="difference-grid"
        style={{ gridTemplateColumns: `repeat(${paired.length},1fr)` }}
      >
        {paired.flatMap((q) =>
          paired.map((k) => {
            const delta =
              (b.attention[q.j]?.[k.j] || 0) - (a.attention[q.i]?.[k.i] || 0);
            return (
              <button
                aria-label={`Attention difference for query ${q.i}, key ${k.i}: ${(delta * 100).toFixed(5)} percentage points`}
                key={`${q.i}-${k.i}`}
                title={`A[${q.i},${k.i}] → B[${q.j},${k.j}]: ${(delta * 100).toFixed(5)} pp`}
                style={{
                  background:
                    delta < 0
                      ? `rgba(230,150,131,${0.08 + 0.92 * Math.abs(delta)})`
                      : `rgba(143,217,201,${0.08 + 0.92 * Math.abs(delta)})`,
                }}
              />
            );
          }),
        )}
      </div>
      <p className="quiet">
        Coral: decrease · teal: increase. Hover for exact differences. Unpaired
        tokens excluded.
      </p>
    </details>
  );
}
function App() {
  const [gallery, setGallery] = useState<Gallery[]>([]),
    [run, setRun] = useState<Run | null>(null),
    [baseline, setBaseline] = useState<Run | null>(null),
    [clean, setClean] = useState<Run | null>(null);
  const [draft, setDraft] = useState(""),
    [layer, setLayer] = useState(0),
    [head, setHead] = useState(0),
    [query, setQuery] = useState(0),
    [tab, setTab] = useState("attention"),
    [all, setAll] = useState(false),
    [busy, setBusy] = useState(""),
    [error, setError] = useState(""),
    [notes, setNotes] = useState(false),
    [settings, setSettings] = useState(false),
    [key, setKey] = useState(sessionStorage.getItem("xray-access") || "");
  const ticket = useRef(0),
    [headData, setHeadData] = useState<Head>(empty),
    [baseHead, setBaseHead] = useState<Head>(empty),
    [hover, setHover] = useState<number | null>(null);
  async function request(path: string, body?: unknown) {
    const response = await fetch(`/api${path}`, {
      method: body ? "POST" : "GET",
      headers: {
        "Content-Type": "application/json",
        "X-Session-ID": session,
        Authorization: `Bearer ${key}`,
      },
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await response.json();
    if (!response.ok)
      throw Error(
        data.detail || data.error || "The model service is unavailable.",
      );
    return data;
  }
  function install(next: Run) {
    setRun(next);
    setDraft(next.prompt);
    setQuery(next.tokens.length - 1);
    setClean(null);
  }
  async function preset(item: Gallery) {
    const id = ++ticket.current;
    setBusy("Opening recorded run…");
    setError("");
    try {
      const result = await fetch(item.file).then((r) => {
        if (!r.ok) throw Error("Could not load the recorded run.");
        return r.json();
      });
      if (id === ticket.current) {
        install({ ...result, ablation: item.ablation });
        setLayer(item.head[0]);
        setHead(item.head[1]);
      }
    } catch (e) {
      if (id === ticket.current) setError(String(e));
    } finally {
      if (id === ticket.current) setBusy("");
    }
  }
  useEffect(() => {
    const close = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setNotes(false);
        setSettings(false);
      }
    };
    document.addEventListener("keydown", close);
    return () => document.removeEventListener("keydown", close);
  }, []);
  useEffect(() => {
    fetch("/data/gallery.json")
      .then((r) => r.json())
      .then((items) => {
        setGallery(items);
        preset(items.find((item:Gallery)=>item.id==="repetition") || items[0]);
      })
      .catch(() =>
        setError("Recorded examples could not load. Reload to retry."),
      );
  }, []);
  useEffect(() => {
    if (!run) return;
    let alive = true;
    setHeadData(empty);
    const local = run.heads?.[layer]?.[head];
    if (local) setHeadData(local);
    else
      request(`/runs/${run.id}/head/${layer}/${head}`)
        .then((data) => {
          if (alive) setHeadData(data);
        })
        .catch((e) => {
          if (alive) setError(String(e));
        });
    return () => {
      alive = false;
    };
  }, [run, layer, head]);
  useEffect(() => {
    if (!baseline) return;
    let alive = true;
    setBaseHead(empty);
    const local = baseline.heads?.[layer]?.[head];
    if (local) setBaseHead(local);
    else
      request(`/runs/${baseline.id}/head/${layer}/${head}`)
        .then((data) => {
          if (alive) setBaseHead(data);
        })
        .catch((e) => {
          if (alive) setError(String(e));
        });
    return () => {
      alive = false;
    };
  }, [baseline, layer, head]);
  async function analyze() {
    if (
      !key &&
      location.hostname !== "127.0.0.1" &&
      location.hostname !== "localhost"
    ) {
      setSettings(true);
      return;
    }
    const id = ++ticket.current;
    setBusy("Waking GPT-2…");
    setError("");
    try {
      const result = await request("/analyze", { prompt: draft });
      if (id === ticket.current) install(result);
    } catch (e) {
      if (id === ticket.current) setError(String(e));
    } finally {
      if (id === ticket.current) setBusy("");
    }
  }
  async function intervene() {
    if (!run) return;
    const id = ++ticket.current;
    setBusy(
      clean
        ? "Restoring clean computation…"
        : "Recomputing with this head disabled…",
    );
    setError("");
    try {
      if (clean) {
        if (run.provenance.source === "recorded") {
          setRun(clean);
          setClean(null);
        } else {
          const result = await request("/analyze", { prompt: clean.prompt });
          if (id === ticket.current) {
            setRun(result);
            setClean(null);
          }
        }
      } else {
        const recorded = gallery.find((g) => g.prompt === run.prompt);
        let result,
          original = run;
        if (run.provenance.source === "recorded") {
          if (
            recorded &&
            recorded.head[0] === layer &&
            recorded.head[1] === head
          )
            result = await fetch(recorded.ablation).then((r) => r.json());
          else if (key) {
            original = await request("/analyze", { prompt: run.prompt });
            result = await request("/intervene", {
              run_id: original.id,
              layer,
              head,
            });
          } else
            throw Error(
              "This example includes one recorded intervention. Connect the live beta to intervene on any head.",
            );
        } else
          result = await request("/intervene", { run_id: run.id, layer, head });
        if (id === ticket.current) {
          setClean(original);
          setBaseline(original);
          setRun(result);
          setTab("compare");
        }
      }
    } catch (e) {
      if (id === ticket.current) setError(String(e));
    } finally {
      if (id === ticket.current) setBusy("");
    }
  }
  const weights = headData.attention[query] || [],
    ranked = weights
      .map((w, k) => ({ w, k }))
      .filter((v) => v.k <= query)
      .sort((a, b) => b.w - a.w),
    edges = all ? ranked : ranked.slice(0, 5),
    omitted = Math.max(0, 1 - edges.reduce((s, e) => s + e.w, 0));
  const alignment = baseline && run ? alignTokens(baseline, run) : [],
    paired = alignment.filter((i: number | null) => i !== null).length;
  const union =
    baseline && run
      ? [
          ...new Set(
            [...baseline.predictions, ...run.predictions].map((t) => t.id),
          ),
        ].map((id) => ({
          id,
          text: [...run.predictions, ...baseline.predictions].find(
            (t) => t.id === id,
          )!.text,
          a:
            baseline.probabilities?.[id] ??
            baseline.predictions.find((t) => t.id === id)?.p ??
            null,
          b:
            run.probabilities?.[id] ??
            run.predictions.find((t) => t.id === id)?.p ??
            null,
        }))
      : [];
  const extent = headData.extent || 1,
    xy = (p: number[]) => [
      220 + (p[0] / extent) * 170,
      200 - (p[1] / extent) * 165,
    ];
  const exportRun = () => {
    if (!run) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(
      new Blob(
        [
          JSON.stringify({
            run,
            baseline,
            selected: { layer, head, query },
            head: headData,
            baselineHead: baseHead,
          }),
        ],
        { type: "application/json" },
      ),
    );
    a.download = `xray-${run.id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
  return (
    <>
      <header>
        <a className="identity" href="/">
          ⌖ <span>Live Prompt X-ray</span>
        </a>
        <div className="header-meta">
          GPT-2 · 124M{" "}
          <button onClick={() => setSettings(true)}>Connect</button>
          <button onClick={() => setNotes(true)}>Notes</button>
        </div>
      </header>
      <main>
        <section className="prompt-bar">
          <div className="eyebrow">A sentence, under the lens.</div>
          <div className="prompt-entry">
            <textarea
              aria-label="Prompt"
              value={draft}
              maxLength={4096}
              onChange={(e) => setDraft(e.target.value)}
              spellCheck={false}
            />
            <button
              className="primary"
              disabled={!!busy || !draft.trim()}
              onClick={analyze}
            >
              Analyze ↗
            </button>
            <button
              disabled={!run || !!busy}
              onClick={() => {
                setBaseline(run);
                setTab("compare");
              }}
            >
              Pin A
            </button>
          </div>
          <div className="presets">
            {gallery.map((item) => (
              <button
                key={item.id}
                className={run?.prompt === item.prompt ? "active" : ""}
                onClick={() => preset(item)}
              >
                {item.title}
              </button>
            ))}
          </div>
          <div className="status" role="status">
            {busy && <i className="activity" />}
            {busy ||
              error ||
              (run?.prompt !== draft
                ? "Draft changed · showing the previous run"
                : run?.provenance.source === "recorded"
                  ? "Recorded example"
                  : "Live observation")}
            {run && (
              <span>
                {run.tokens.length} tokens ·{" "}
                {run.intervention ? "head disabled" : "clean run"}
              </span>
            )}
          </div>
        </section>
        {run && (
          <>
            <div className="tokens" aria-label="Choose a query token">
              {run.tokens.map((t, i) => (
                <button
                  key={i}
                  className={`${query === i ? "selected" : ""} ${t.boundary ? "boundary" : ""}`}
                  title={`Position ${i} · ID ${t.id}`}
                  aria-pressed={query === i}
                  onClick={() => setQuery(i)}
                >
                  <small>{String(i).padStart(2, "0")}</small>
                  {t.boundary ? "BOS" : show(t.text)}
                </button>
              ))}
            </div>
            <div className="workspace">
              <aside className="heads">
                <div className="section-label">
                  Heads <span>concentration</span>
                </div>
                <div className="head-labels">
                  <span />{" "}
                  {Array.from({ length: 12 }, (_, i) => (
                    <small key={i}>{i + 1}</small>
                  ))}
                </div>
                {run.concentration.map((row, l) => (
                  <div className="head-row" key={l}>
                    <small>{l + 1}</small>
                    {row.map((values, h) => (
                      <button
                        key={h}
                        title={`Layer ${l + 1}, head ${h + 1} · ${pct(values[query] || 0)} concentration`}
                        aria-label={`Layer ${l + 1}, head ${h + 1}`}
                        aria-pressed={l === layer && h === head}
                        style={{
                          background: `rgba(143,217,201,${0.06 + 0.94 * (values[query] || 0)})`,
                        }}
                        onClick={() => {
                          setLayer(l);
                          setHead(h);
                        }}
                      />
                    ))}
                  </div>
                ))}
                <div className="head-selection">
                  <span>Layer {layer + 1}</span>
                  <span>Head {head + 1}</span>
                </div>
                <button
                  className="intervene"
                  disabled={!!busy}
                  onClick={intervene}
                >
                  {clean ? "Restore head" : "Disable this head"} ↗
                </button>
                <p className="quiet">
                  {run.intervention
                    ? "Zero-ablation · same input"
                    : "Select a head to inspect it."}
                </p>
              </aside>
              <section className="visual">
                <nav>
                  {["attention", "query/key map", "compare"].map((t) => (
                    <button
                      key={t}
                      className={tab === t ? "active" : ""}
                      onClick={() => setTab(t)}
                    >
                      {t === "query/key map"
                        ? "Query / key"
                        : t[0].toUpperCase() + t.slice(1)}
                    </button>
                  ))}
                </nav>
                {tab === "attention" && (
                  <>
                    <div className="attention-views">
                      <Matrix
                        head={headData}
                        query={query}
                        onQuery={setQuery}
                        label="Measured attention matrix"
                      />
                      <div className="arc-panel">
                        <svg
                          viewBox="0 0 460 240"
                          role="img"
                          aria-label="Strongest attention edges"
                        >
                          {edges.map(({ w, k }) => {
                            const x =
                                24 +
                                (k / Math.max(1, run.tokens.length - 1)) * 410,
                              source =
                                24 +
                                (query / Math.max(1, run.tokens.length - 1)) *
                                  410;
                            return (
                              <g key={k}>
                                <path
                                  d={`M${source},190 Q${(source + x) / 2},${25 + 60 * (1 - w)} ${x},190`}
                                  fill="none"
                                  stroke="#8fd9c9"
                                  strokeWidth={1 + 2 * Math.sqrt(w)}
                                  opacity={0.2 + 0.8 * Math.sqrt(w)}
                                />
                                <text
                                  x={x}
                                  y={211 + (k % 2) * 15}
                                  fill="#afbdc0"
                                  textAnchor="middle"
                                  fontSize="10"
                                >
                                  {k}
                                </text>
                              </g>
                            );
                          })}
                          <circle
                            cx={
                              24 +
                              (query / Math.max(1, run.tokens.length - 1)) * 410
                            }
                            cy="190"
                            r="4"
                            fill="#e5b985"
                          />
                        </svg>
                        <div className="edge-list">
                          {edges.map(({ w, k }) => (
                            <button
                              key={k}
                              onClick={() => setHover(k)}
                              className={hover === k ? "active" : ""}
                            >
                              <span>
                                {k} · {show(run.tokens[k].text)}
                              </span>
                              <b>{pct(w)}</b>
                            </button>
                          ))}
                        </div>
                        <button
                          className="text-button"
                          onClick={() => setAll(!all)}
                        >
                          {all ? "Strongest five" : "Show all edges"} ·{" "}
                          {pct(omitted)} omitted
                        </button>
                      </div>
                    </div>
                  </>
                )}
                {tab === "query/key map" && (
                  <>
                    <svg
                      className="cloud"
                      viewBox="0 0 440 420"
                      role="img"
                      aria-label="Queries and keys projected on a fixed calibrated PCA basis"
                    >
                      <path d="M30 200H410M220 25V375" stroke="#2b3b42" />
                      {headData.k.map((p, i) => {
                        const [x, y] = xy(p);
                        return (
                          <g key={i} onMouseEnter={() => setHover(i)}>
                            <rect
                              x={x - 3}
                              y={y - 3}
                              width="6"
                              height="6"
                              fill={i > query ? "#536169" : "#e69683"}
                            />
                            <text x={x + 5} y={y} fontSize="9" fill="#87999f">
                              {i}
                            </text>
                          </g>
                        );
                      })}
                      {headData.q.map((p, i) => {
                        const [x, y] = xy(p);
                        return (
                          <circle
                            key={i}
                            cx={x}
                            cy={y}
                            r={i === query ? 6 : 3}
                            fill="#8fd9c9"
                            opacity={i === query ? 1 : 0.35}
                            onClick={() => setQuery(i)}
                          />
                        );
                      })}
                    </svg>
                    <div className="plot-foot">
                      ● Query · ■ Key · fixed PCA coordinates
                    </div>
                    <p className="quiet">
                      {hover !== null
                        ? `Query ${query} → key ${hover}: ${hover > query ? "masked" : pct(weights[hover] || 0)}`
                        : "Hover a key for its measured attention weight. Distance alone is not attention."}
                    </p>
                  </>
                )}
                {tab === "compare" &&
                  (baseline ? (
                    <>
                      <div className="compare-heading">
                        <span>
                          A · {baseline.intervention ? "ablated" : "pinned"}
                        </span>
                        <span>
                          B · {run.intervention ? "zero-ablation" : "current"}
                        </span>
                      </div>
                      <div className="compare-prompts">
                        <p>{baseline.prompt}</p>
                        <p>{run.prompt}</p>
                      </div>
                      <div className="compare-matrices">
                        <Matrix
                          head={baseHead}
                          query={Math.min(query, baseline.tokens.length - 1)}
                          onQuery={() => {}}
                          label="Complete baseline attention matrix"
                        />
                        <Matrix
                          head={headData}
                          query={query}
                          onQuery={setQuery}
                          label="Complete current attention matrix"
                        />
                      </div>
                      <p className="quiet">
                        {paired} / {baseline.tokens.length} baseline tokens
                        unambiguously paired. Insertions, deletions and
                        ambiguous spans stay unpaired.
                      </p>
                      <Difference
                        a={baseHead}
                        b={headData}
                        alignment={alignment}
                      />
                      <div className="delta-list">
                        {union.map((t) => (
                          <div key={t.id}>
                            <span>{show(t.text)}</span>
                            <span>
                              {t.a === null ? "outside top 10" : pct(t.a)}
                            </span>
                            <span>
                              {t.b === null ? "outside top 10" : pct(t.b)}
                            </span>
                            <i className="delta-axis">
                              <em
                                style={{
                                  left: `${50 + Math.min(0, ((t.b || 0) - (t.a || 0)) * 50)}%`,
                                  width: `${Math.abs((t.b || 0) - (t.a || 0)) * 50}%`,
                                  background:
                                    (t.b || 0) >= (t.a || 0)
                                      ? "#8fd9c9"
                                      : "#e69683",
                                }}
                              />
                            </i>
                            <b>
                              {t.a !== null && t.b !== null
                                ? `${((t.b - t.a) * 100).toFixed(2)} pp`
                                : "—"}
                            </b>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="empty">
                      <h2>Keep a point of reference.</h2>
                      <p>Pin A, change the prompt, then analyze B.</p>
                    </div>
                  ))}
              </section>
              <aside className="predictions">
                <div className="section-label">
                  Next token <span>final position</span>
                </div>
                <div className="winner">
                  {show(run.predictions[0]?.text || "")}
                  <small>{pct(run.predictions[0]?.p || 0)}</small>
                </div>
                {run.predictions.map((t) => (
                  <div
                    className="prediction"
                    key={t.id}
                    title={`Token ${t.id} · ${t.p.toPrecision(8)}`}
                  >
                    <span>{show(t.text)}</span>
                    <i>
                      <b
                        style={{
                          width: `${(t.p / (run.predictions[0]?.p || 1)) * 100}%`,
                        }}
                      />
                    </i>
                    <small>{pct(t.p)}</small>
                  </div>
                ))}
                <div className="run-note">
                  <span>
                    {run.intervention ? "Intervention" : "Observation"}
                  </span>
                  <code>{run.id.slice(0, 12)}</code>
                  <button onClick={exportRun}>Download experiment ↓</button>
                  <button
                    onClick={() => saveCard(run, headData, layer, head, query)}
                  >
                    Save image ↓
                  </button>
                </div>
              </aside>
            </div>
          </>
        )}
      </main>
      {(notes || settings) && (
        <div
          className="modal-backdrop"
          onClick={() => {
            setNotes(false);
            setSettings(false);
          }}
        >
          <section
            role="dialog"
            aria-modal="true"
            aria-label={notes ? "Experiment notes" : "Live connection"}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="close"
              autoFocus
              onClick={() => {
                setNotes(false);
                setSettings(false);
              }}
            >
              Close ×
            </button>
            {notes ? (
              <>
                <h2>What the lens shows.</h2>
                <p>
                  Measured attention, projected queries and keys, and next-token
                  probabilities from GPT-2. Attention is a weighted aggregation,
                  not a complete explanation of an answer.
                </p>
                <p>
                  One explicit boundary token precedes your exact text. Heads
                  are colored by concentration for your selected query.
                  Predictions always belong to the final prompt position.
                </p>
                <p>
                  The query/key map uses a frozen shared PCA basis per head.
                  Projection distances are approximate. Disabling a head zeros
                  its output before the attention projection, across every
                  position; it does not erase the attention matrix.
                </p>
                <p>
                  Gallery examples are saved executions. Each default head was
                  selected for the largest drop in the clean top-token
                  probability across all 144 single-head ablations. Live input
                  is session-scoped and expires after ten minutes.
                </p>
                <pre>{JSON.stringify(run?.provenance, null, 2)}</pre>
                <p>
                  Built with GPT-2 and TransformerLens. Inspired by
                  AttentionViz.
                </p>
              </>
            ) : (
              <>
                <h2>Live connection</h2>
                <p>
                  The recorded gallery works without compute. The private live
                  beta requires an access token.
                </p>
                <label>
                  Access token
                  <input
                    type="password"
                    value={key}
                    onChange={(e) => setKey(e.target.value)}
                  />
                </label>
                <button
                  className="primary"
                  onClick={() => {
                    sessionStorage.setItem("xray-access", key);
                    setSettings(false);
                  }}
                >
                  Save for this tab
                </button>
              </>
            )}
          </section>
        </div>
      )}
    </>
  );
}
createRoot(document.getElementById("root")!).render(<App />);

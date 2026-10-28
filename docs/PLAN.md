# Live Prompt X-ray

Status: a standalone implementation plan, saved before implementation. This is independent of Transformer Time Machine and Cinematic 3D. Prepared October 25, 2026.

## The experience

Type a sentence, select a token, and explore the actual attention patterns and next-token predictions of a small language model. Pin the sentence, change a word, and compare the two runs. Disable one attention head and see what changes in the output.

The audience is curious technical readers who understand that models process tokens but have never inspected one interactively. The finished artifact is a polished, reproducible visual instrument and a short screen recording. It does not need a scientific discovery to succeed. The promise is “edit a prompt and inspect measured internal changes,” not “read the model’s thoughts.”

The default scene uses a short repeated sequence to make a recognizable attention pattern visible. Five additional presets cover a name substitution, a repeated phrase, punctuation, a typo, and a short natural-language completion. Select these for legibility after inspecting real runs; retain examples with weak or unexpected patterns rather than inventing a textbook outcome.

The principal interaction takes under a minute:

1. Load a preset or enter text and press **Analyze**.
2. Select a query token; the head mosaic, attention matrix, and token arcs update together.
3. Select a head to inspect its exact weights and query/key geometry.
4. Pin baseline A, edit text into B, and compare predictions and aligned token relationships.
5. Press **Disable this head** to run a clearly labeled intervention, then restore it.

## Scope and visual design

Build a desktop-first React/TypeScript interface with Canvas for dense plots and SVG/HTML for labels, arcs, and controls. Use a restrained dark background, high-contrast text, teal queries, coral keys, and one shared sequential scale for attention strength. Color must have a textual or shape equivalent. Avoid decorative moving particles: animation only shows selection transitions or completed-run changes. Provide reduced motion and keyboard navigation.

Arrange the workspace into a prompt/token strip, central visualization, and evidence panel. The central tabs are **Attention**, **Query/key map**, and **Compare**. Keep the 12-layer-by-12-head mosaic visible alongside them. Its default color represents attention entropy for the selected query, explicitly labeled as concentration, not importance. Hover exposes the value and layer/head indices. Attention arcs show the five strongest eligible keys, with an option to reveal all; omitted attention mass remains visible.

Show the top ten next-token candidates at the final prompt position, plus exact probability for a selected candidate. Inspecting earlier query tokens changes the attention view without silently changing the prediction position. Display token IDs, whitespace markers, position, model name, and run state on demand. Boundary tokens are visually separate from user input.

V1 supports one model, one pinned comparison, at most 256 total input tokens including the fixed boundary token, and one head intervention at a time. Reject overlength input with the token count; do not silently truncate. Free-running generation, model training, automatic circuit discovery, checkpoint timelines, and photorealistic 3D are outside this project.

## Model, tensors, and implementation

Use the 124M-parameter **openai-community/gpt2** checkpoint in evaluation mode, float32, with explicit eager attention. Its small size and familiar architecture make it suitable for interactive inspection; it is an English completion model, not a modern conversational assistant. Use its matching fast tokenizer and prepend exactly one EOS/BOS ID according to an explicit application policy; disable automatic duplication and record the policy.

Use the local Hugging Face execution driver through TransformerLens’s current `TransformerBridge.boot_transformers` interface. The current documentation demonstrates GPT-2 and supports cached activations and temporary hooks. Do not assume old `HookedTransformer` examples or serving/remote drivers provide equivalent mutable hooks. Before UI work, resolve and lock exact model/tokenizer revisions, TransformerLens commit or release, Transformers, PyTorch, and the container image digest. Verify the pinned GPT-2 adapter exposes the required tensors and that interventions change the actual forward computation. If that gate fails, fix the adapter in the pinned stack; do not ship guessed tensors or quietly substitute another model.

The Python inference service collects actual attention probabilities `[layer, head, query, key]`, query/key vectors, and final-position logits. Cache only required tensors, detach them, and release GPU references after each response is assembled. Keep model execution serialized per worker so temporary hooks cannot leak between requests. Always remove intervention hooks in a `finally` path.

For the selected head, project queries and keys onto a common two-dimensional PCA basis fitted once to a frozen, published calibration-prompt set. Use one joint mean and basis, deterministic component signs, and store the fitted transform per head. Reuse it across edited prompts so dots do not move simply because a new embedding was fitted. Axes are projection coordinates, not semantic dimensions. Proximity is approximate: hovering always reveals the measured attention edge, and masked future keys never appear as valid attention destinations. Do not alter model Q/K values to make prettier clusters.

Use a FastAPI service deployed on Modal and a static frontend. Define `POST /api/analyze`, `GET /api/runs/{id}/head/{layer}/{head}`, and `POST /api/intervene`. Analysis returns token metadata, final-position predictions, head summaries, provenance, and a session-scoped run ID. Fetch a selected head’s matrix and projected coordinates on demand as compressed numeric arrays. Intervention accepts that run ID and one valid layer/head pair, returning an independently identified result with the same provenance plus the intervention specification. Error responses distinguish invalid input, expired run, capacity limit, model failure, and unavailable service.

Cache up to eight runs per session for ten minutes in worker memory. A cache miss after scale-down triggers an explicit rerun, not a fabricated lookup. Preserve the last completed visualization while a request runs, mark it as stale if text changes, and discard out-of-order responses. An **Analyze** button controls compute; typing never fires a GPU request.

## Comparison and intervention semantics

Tokenize A and B independently. Align unchanged character spans using a deterministic text diff, then map tokenizer offsets into those spans. Pair tokens only when their character spans, exact text, and token IDs match unambiguously. Highlight inserted/deleted/resegmented spans and leave their cells unpaired. Do not align by token index or by decoded token text alone: repeated words and byte-level tokenization make that misleading. Display each version’s complete matrix; a difference overlay includes only explicitly matched query/key pairs. This is a descriptive comparison, not an isolated causal effect of changing a word.

**Disable this head** zeros that head’s attention output before its contribution is combined into the residual stream, at every position in the prompt. It does not zero the attention probabilities and renormalize other heads. Recompute the full forward pass on the identical token IDs. Show baseline/intervention probability deltas for the union of their top-ten candidate tokens, with a shared axis. Call the result a **zero-ablation effect on this prompt**. It does not prove the head alone represents a concept, is universally necessary, or has no redundant substitutes. Restoration reruns the clean path. Cross-prompt activation patching and automatic head rankings are excluded from v1.

## Provenance, compute, and rollout

Each downloadable experiment bundle contains exact input text/token IDs, normalization and boundary policy, checkpoint/tokenizer revisions, execution-library versions, dtype/device, tensor names/shapes, projection transform hash, intervention settings, timestamps, and measured latency. Include machine-readable arrays and a compact screenshot with the run ID. The server does not retain custom prompts after session expiry or publish them automatically. Saved examples are explicitly curated public fixtures; live sessions remain session-scoped.

First run a CPU compatibility check. Then reserve **at most $25 of Modal credits** for a GPU pilot on one L4, one concurrent worker, 60-second idle shutdown, 120-second per-call timeout, and a maximum of 50 requests spanning short and maximum-length inputs. Count allocated GPU, CPU, and memory time toward the cap and stop dispatch before the next bounded request can exceed it. Record warm/cold latency, peak memory, payload size, cache behavior, and cost per successful interaction. These are limits, not claims about expected runtime.

Do not scale to H100 merely to consume credits. After the pilot, keep further GPU work paused until its measured cost is reviewed. The release target is a warm p95 below three seconds for 128-token prompts and below six seconds for 256 tokens, measured over at least 30 requests; these are acceptance targets, not promises. If missed, profile extraction and transfer first. Show cold starts honestly.

Ship a static gallery that works without a GPU, then an unlisted live beta protected by a server-side access token, with one request per session at a time and at most two workers. Public live access is a separate decision based on measured cost. No scheduled GPU keepalive is required.

## Milestones and acceptance

1. **Verified extraction:** locked environment, reproducible fixtures, successful instrumentation gate, and pilot cost report.
2. **Single-run explorer:** linked token/mosaic/matrix/cloud views with real arrays, loading/errors, accessibility, and source metadata.
3. **Edit and intervene:** aligned A/B comparisons, measured head ablation, restoration, session isolation, and export.
4. **Release artifact:** six verified presets, static fallback, unlisted beta, setup/reproduction instructions, and a 30–60-second screen recording using actual results.

Required checks:

- Instrumented clean logits match the uninstrumented eager GPT-2 path within documented float32 tolerances; fixed fixtures preserve top predictions.
- Attention rows sum to one within tolerance, future-position entries are zero, and displayed values match captured tensor cells.
- Captured pre-projection head output reconstructs the expected attention-output contribution; zeroing only the selected head produces the registered intervention. Restoration matches baseline.
- Leading spaces, Unicode, repeated words, insertions, deletions, and token splits produce correct or explicitly unpaired alignments.
- Projection reuse is deterministic; nearest dots are never presented as verified high-attention pairs without measured weights.
- Expiry, cold start, timeout, duplicate clicks, stale responses, and concurrent users cannot mix inputs or leave hooks installed.
- A fresh installation reproduces a saved fixture without any frontend fixture pretending to be a live inference result.

## Sources and future choices

Verified primary references: [GPT-2 model card](https://huggingface.co/openai-community/gpt2), [TransformerLens loading guide](https://transformerlensorg.github.io/TransformerLens/content/getting_started.html), [migration and driver limitations](https://transformerlensorg.github.io/TransformerLens/content/migrating_to_v3.html), and [AttentionViz](https://arxiv.org/abs/2305.03210). The supplied `Paper (4).pdf`, pp. 1–4, motivates linked Q/K inspection; `Exploring Induction Heads.pdf`, pp. 2–6, motivates repeated-token presets and head mosaics. Their observed attention patterns are not treated as causal findings.

Default assumptions: no model training, no novelty claim, no closed-model introspection, and no dependency on the other two planned projects. Later options—each separately scoped—include a larger modern model, multi-token continuation, controlled activation patching, learned-feature views, or a carefully evaluated nonlinear projection.

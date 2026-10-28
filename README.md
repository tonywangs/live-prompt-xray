# Live Prompt X-ray

Look inside one GPT-2 prediction. Select tokens and heads, inspect attention and fixed-basis query/key projections, pin a baseline, edit the input, and zero-ablate a head to measure its effect.

## Run

```sh
npm ci
npm run dev
```

The six recorded examples work immediately. Each includes the complete 12×12 head set, exact output probabilities, and one recorded intervention. Default intervention heads were selected by exhaustively evaluating all 144 single-head ablations and choosing the greatest decrease in the clean top-token probability. This is a deliberately selected demonstration, not an unbiased estimate of typical head importance. All scores are in `docs/intervention-selection.json`.

For arbitrary prompts and head interventions:

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
export XRAY_ACCESS_TOKEN='your-private-token'
.venv/bin/uvicorn backend.api:app --host 127.0.0.1 --port 8001
```

Open **Connect** and enter that token. Vite proxies `/api` to the local service. Text is preserved exactly; a single EOS/BOS token is prepended. Maximum context: 256 tokens including that boundary token. Inference happens only when you press Analyze.

## Deploy

The frontend deploys to Vercel. The `/api` rewrite in `vercel.json` points to the Modal endpoint and forwards the private access token; no secret is embedded in the browser bundle. Change that destination when deploying your own backend. Deploy the backend with `python backend/deploy.py` after authenticating Modal. It creates an ignored, permission-restricted `.env.local` and a Modal secret. Do not commit or share that file.

The CPU service serializes computation, allows at most four concurrent sessions, caches eight runs per session for ten minutes, and caps the first pilot at **50 attempted model runs across worker restarts** using a Modal dictionary. It has one worker, a 120-second timeout and 60-second idle scale-down. Review usage before explicitly resetting the pilot counter. Public visitors can use the recorded gallery without invoking compute. No GPU has been allocated.

## Reproduce and verify

```sh
python -m backend.engine
python -m backend.select_interventions
python -m backend.check
npm test
npm run build
```

Model and tokenizer: `openai-community/gpt2`, immutable revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. PyTorch 2.14.1, Transformers 5.21.0, TransformerLens 4.3.0; float32, eager attention, evaluation mode, no KV cache. Native reference logits gate startup: maximum absolute difference <0.0003 and identical argmax at every position of eight calibration fixtures. The observed local maximum is 0.000205994. TransformerLens's bridge decomposes native operations, so float32 round-off is not bitwise identical. The stricter 0.0001 single-prompt check passed, but did not generalize to the full calibration set; the documented gate reflects that measured difference. Native Pythia parity in the separate cinematic project is exact.

The frozen joint Q/K PCA basis, means, calibration texts, viewport extent and SHA-256 live in `public/data/projection.json`. Axes do not refit when a prompt changes. A projected distance is not an attention score; causal masking and the full dot product still matter. Out-of-calibration coordinates may extend beyond the fixed viewport.

Head ablation zeros the selected head output before the output projection at every position. It recomputes downstream operations; it does not merely hide the matrix. Restoring produces exactly the clean probabilities. Hooks are removed by TransformerLens's scoped execution. Attention remains visible for an ablated head because its output, not its softmax weights, is disabled.

A/B pairing requires unambiguous character correspondences across all longest common subsequences, matching token IDs and exact spans. Unicode offsets are converted to browser UTF-16 indices. Unpaired tokens remain visible in full matrices and are omitted from the difference grid. Predictions always refer to the final input position, independently of the selected query.

Local API tests cover authorization, session isolation, expiry, eviction, token limits, Unicode offsets, ablation and restoration. Thirty warm runs at each length measured local CPU p95 ≈0.19s at 128 tokens and ≈0.28s at 256; these exclude network and cold-start time and are not hosted service guarantees. See `docs/api-validation.json` and `docs/remote-validation.json`.

Download experiment JSON or a PNG evidence card from the interface. No prompts are deliberately logged or written to disk by the app; hosting infrastructure may retain ordinary request metadata.

## Attribution

[GPT-2](https://huggingface.co/openai-community/gpt2), [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), [AttentionViz](https://arxiv.org/abs/2305.03210), and [Cynthia Chen's transformer experiments](https://github.com/chenxcynthia/transformers). Original interface and code. GPT-2's MIT license applies to its model assets; no model weights are distributed here.

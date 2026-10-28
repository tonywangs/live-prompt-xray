import json, torch
from pathlib import Path
from .engine import Engine

engine = Engine()
ids, _ = engine.ids(" red blue green red blue green red")
collected = {}
hooks = []
for l in range(12):
    for suffix in ["hook_pattern", "v.hook_out", "o.hook_in"]:
        name = f"blocks.{l}.attn.{suffix}"

        def save(v, hook, name=name):
            collected[name] = v.detach().clone()

        hooks.append((name, save))
with torch.inference_mode():
    engine.bridge.run_with_hooks(ids, fwd_hooks=hooks, use_cache=False)
errors = []
for l in range(12):
    p = f"blocks.{l}.attn."
    actual = collected[p + "o.hook_in"]
    expected = (
        collected[p + "hook_pattern"] @ collected[p + "v.hook_out"].transpose(1, 2)
    ).transpose(1, 2)
    errors.append(float((actual - expected).abs().max()))
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
_, ablated = engine.capture(ids, (1, 3))
assert torch.count_nonzero(ablated["blocks.1.attn.o.hook_in"][:, :, 3]) == 0
Path(__file__).resolve().parents[1].joinpath(
    "docs/head-math-validation.json"
).write_text(
    json.dumps(
        {"weighted_values_max_abs_by_layer": errors, "selected_head_output_zero": True},
        indent=2,
    )
)
print("All 12 layers reconstruct; selected head is actually zero.")

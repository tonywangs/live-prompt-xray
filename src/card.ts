import type { Run, Head } from "./types";
export function saveCard(
  run: Run,
  head: Head,
  layer: number,
  h: number,
  query: number,
) {
  const c = document.createElement("canvas");
  c.width = 1400;
  c.height = 900;
  const x = c.getContext("2d")!;
  x.fillStyle = "#0c1419";
  x.fillRect(0, 0, 1400, 900);
  x.fillStyle = "#dfebe8";
  x.font = "30px Georgia";
  x.fillText("Live Prompt X-ray", 50, 60);
  x.font = "20px monospace";
  const lines = run.prompt.match(/.{1,95}/g) || [];
  lines.slice(0, 3).forEach((line, i) => x.fillText(line, 50, 115 + i * 30));
  x.fillStyle = "#94a5aa";
  x.font = "15px monospace";
  x.fillText(
    `Layer ${layer + 1} · Head ${h + 1} · Query ${query} · ${run.intervention ? "head zero-ablated" : "clean computation"}`,
    50,
    235,
  );
  const n = head.attention.length,
    size = 480 / n;
  head.attention.forEach((row, q) =>
    row.forEach((v, k) => {
      x.fillStyle =
        k > q
          ? "#172126"
          : `rgb(${32 + 98 * Math.sqrt(v)},${49 + 168 * Math.sqrt(v)},${52 + 151 * Math.sqrt(v)})`;
      x.fillRect(50 + k * size, 280 + q * size, size - 1, size - 1);
    }),
  );
  x.strokeStyle = "#e6b87b";
  x.strokeRect(49, 279 + query * size, 482, size + 2);
  x.fillStyle = "#94a5aa";
  x.fillText("Next token · final prompt position", 650, 280);
  run.predictions.forEach((p, i) => {
    x.fillStyle = "#dfebe8";
    x.fillText(p.text.replaceAll(" ", "·"), 650, 325 + i * 40);
    x.fillStyle = "#8fd9c9";
    x.fillRect(880, 310 + i * 40, (p.p / run.predictions[0].p) * 300, 4);
    x.fillText(`${(p.p * 100).toFixed(3)}%`, 1210, 325 + i * 40);
  });
  x.fillStyle = "#94a5aa";
  x.font = "13px monospace";
  x.fillText(`GPT-2 · ${run.provenance.model_revision}`, 50, 825);
  x.fillText(
    "Attention is weighted aggregation, not a complete explanation. Raw data available in experiment JSON.",
    50,
    855,
  );
  c.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob!);
    a.download = `xray-${run.id}.png`;
    a.click();
    URL.revokeObjectURL(a.href);
  });
}

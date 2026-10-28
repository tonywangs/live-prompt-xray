"""Exhaustively score 144 heads per gallery prompt; no invented causal claims."""
import json
from pathlib import Path
from .engine import Engine,PRESETS
root=Path(__file__).resolve().parents[1];e=Engine();out=root/'public/data';gallery=[];report=[]
for slug,title,prompt,_,_ in PRESETS:
 ids,_=e.ids(prompt);clean=e.run(prompt,source='recorded');target=clean['predictions'][0]['id'];scores=[]
 for layer in range(12):
  for head in range(12):
   logits,_=e.capture(ids,(layer,head));p=float(logits[0,-1].double().softmax(-1)[target]);scores.append({'layer':layer,'head':head,'probability_drop':clean['probabilities'][target]-p})
 best=max(scores,key=lambda r:r['probability_drop']);l,h=best['layer'],best['head'];ablated=e.run(prompt,(l,h),source='recorded');assert e.run(prompt,source='recorded')['probabilities']==clean['probabilities']
 for suffix,run in [('',clean),('-ablated',ablated)]: (out/f'{slug}{suffix}.json').write_text(json.dumps(run,separators=(',',':')))
 gallery.append({'id':slug,'title':title,'prompt':prompt,'file':f'/data/{slug}.json','ablation':f'/data/{slug}-ablated.json','head':[l,h]});report.append({'prompt':prompt,'selection':'largest decrease in clean top-token probability among all 144 single-head ablations','selected':best,'scores':scores});print(slug,best,flush=True)
(out/'gallery.json').write_text(json.dumps(gallery));(root/'docs/intervention-selection.json').write_text(json.dumps(report,indent=2))

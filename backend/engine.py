"""Pinned GPT-2 observations. Native reference parity gates every engine startup."""
import hashlib,json,time,uuid,threading
from pathlib import Path
import numpy as np
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformer_lens.model_bridge import TransformerBridge
MODEL='openai-community/gpt2'
REVISION='607a30d783dfa663caf39e06633721c8d4cfcd7e'
CALIBRATION=['The capital of France is Paris.','Once upon a time, a little fox crossed the river.','Alice gave Bob a book. Bob thanked Alice.',' red blue green red blue green','def add(a, b):\n    return a + b','A telescope gathers light from distant stars.','One, two, three. One, two, three.','The dog chased the cat, but the cat escaped.']
PRESETS=[('completion','A familiar beginning','Once upon a',2,6),('names','Names & references','Alice gave Bob a book. Bob thanked',9,9),('repetition','A repeating pattern',' red blue green red blue green red',5,1),('phrase','A returning phrase','The rain fell softly. The rain fell',8,6),('punctuation','A small pause','Wait, wait... What happens next?',6,4),('typo','An imperfect sentence','The captial of France is',8,7)]
class Engine:
 def __init__(self):
  torch.set_num_threads(4);self.lock=threading.RLock()
  self.tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
  model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float32,attn_implementation='eager').eval()
  probes=[self.ids(s)[0] for s in CALIBRATION]
  with torch.inference_mode(): references=[model(i,use_cache=False).logits.clone() for i in probes]
  self.bridge=TransformerBridge.boot_transformers(MODEL,hf_model=model,tokenizer=self.tokenizer,device='cpu',dtype=torch.float32,revision=REVISION)
  with torch.inference_mode(): outputs=[self.bridge(i,use_cache=False) for i in probes]
  self.parity=max(float((a-b).abs().max()) for a,b in zip(references,outputs))
  assert self.parity<3e-4, self.parity
  assert all(torch.equal(a.argmax(-1),b.argmax(-1)) for a,b in zip(references,outputs))
  self.basis=[]
  frozen=Path(__file__).resolve().parents[1]/'public/data/projection.json'
  if frozen.exists():
   saved=json.loads(frozen.read_text());self.basis=saved['heads'];self.basis_hash=hashlib.sha256(json.dumps(self.basis,sort_keys=True).encode()).hexdigest();assert self.basis_hash==saved['sha256'];return
  caches=[self.capture(i)[1] for i in probes]
  for l in range(12):
   row=[]
   for h in range(12):
    x=np.concatenate([c[f'blocks.{l}.attn.{kind}.hook_out'][0,:,h].numpy() for c in caches for kind in ['q','k']]);mean=x.mean(0);_,_,v=np.linalg.svd(x-mean,full_matrices=False);basis=v[:2].T
    for j in range(2):
     if basis[np.abs(basis[:,j]).argmax(),j]<0:basis[:,j]*=-1
    extent=float(np.abs((x-mean)@basis).max())*1.15
    row.append({'mean':mean.tolist(),'basis':basis.tolist(),'extent':extent})
   self.basis.append(row)
  self.basis_hash=hashlib.sha256(json.dumps(self.basis,sort_keys=True).encode()).hexdigest()
 def ids(self,text):
  enc=self.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
  ids=torch.tensor([[self.tokenizer.eos_token_id]+enc['input_ids']]);assert ids.shape[1]<=256,'Limit: 255 text tokens plus one boundary token.'
  utf=lambda n:len(text[:n].encode('utf-16-le'))//2
  tokens=[{'id':self.tokenizer.eos_token_id,'text':'<|endoftext|>','offset':[0,0],'boundary':True}]+[{'id':i,'text':self.tokenizer.decode([i]),'offset':[utf(a),utf(b)]} for i,(a,b) in zip(enc['input_ids'],enc['offset_mapping'])]
  return ids,tokens
 def capture(self,ids,ablation=None):
  cache={};hooks=[]
  for l in range(12):
   for suffix in ['q.hook_out','k.hook_out','hook_pattern','o.hook_in']:
    name=f'blocks.{l}.attn.{suffix}'
    def save(value,hook,name=name):cache[name]=value.detach().clone()
    if ablation and l==ablation[0] and suffix=='o.hook_in':
     def zero(value,hook):
      value=value.clone();value[:,:,ablation[1]]=0;return value
     hooks.append((name,zero))
    hooks.append((name,save))
  with torch.inference_mode(): logits=self.bridge.run_with_hooks(ids,fwd_hooks=hooks,use_cache=False)
  return logits,cache
 def run(self,prompt,ablation=None,source='live',compact=False):
  with self.lock:
   start=time.perf_counter();ids,tokens=self.ids(prompt);logits,c=self.capture(ids,ablation);prob=logits[0,-1].double().softmax(-1);p,ix=prob.topk(10);heads=[];concentration=[]
   for l in range(12):
    hs=[];cs=[]
    for h in range(12):
     a=c[f'blocks.{l}.attn.hook_pattern'][0,h];assert float((a.sum(-1)-1).abs().max())<1e-5;assert float(a.triu(1).abs().max())==0
     basis=self.basis[l][h];mean=np.array(basis['mean']);b=np.array(basis['basis'])
     project=lambda kind:((c[f'blocks.{l}.attn.{kind}.hook_out'][0,:,h].numpy()-mean)@b).tolist()
     hs.append({'attention':a.numpy().copy() if compact else a.tolist(),'q':project('q'),'k':project('k'),'extent':basis['extent']})
     cs.append((a*a).sum(-1).tolist())
    heads.append(hs);concentration.append(cs)
   result={'id':str(uuid.uuid4()),'prompt':prompt,'tokens':tokens,'heads':heads,'concentration':concentration,'predictions':[{'id':int(i),'text':self.tokenizer.decode([i]),'p':float(v)} for i,v in zip(ix,p)],'probabilities':prob.tolist(),'provenance':{'model':MODEL,'model_revision':REVISION,'tokenizer_revision':REVISION,'dtype':'float32','device':'cpu','torch':torch.__version__,'transformers':transformers.__version__,'transformer_lens':'4.3.0','attention':'eager','boundary':'one explicit EOS/BOS','source':source,'parity_max_abs':self.parity,'pca_sha256':self.basis_hash,'latency_seconds':time.perf_counter()-start,'head_concentration':'sum of squared attention weights','intervention':'zero head output before projection at every position' if ablation else None}}
   if ablation:result['intervention']={'layer':ablation[0],'head':ablation[1]}
   return result
if __name__=='__main__':
 root=Path(__file__).resolve().parents[1];out=root/'public/data';out.mkdir(parents=True,exist_ok=True);engine=Engine();gallery=[];checks=[]
 for id,title,prompt,l,h in PRESETS:
  clean=engine.run(prompt,source='recorded');ablated=engine.run(prompt,(l,h),source='recorded');restored=engine.run(prompt,source='recorded');assert clean['probabilities']==restored['probabilities'];assert clean['probabilities']!=ablated['probabilities']
  for suffix,data in [('',clean),('-ablated',ablated)]: (out/f'{id}{suffix}.json').write_text(json.dumps(data,separators=(',',':')))
  gallery.append({'id':id,'title':title,'prompt':prompt,'file':f'/data/{id}.json','ablation':f'/data/{id}-ablated.json','head':[l,h]});checks.append({'prompt':prompt,'restore_exact':True,'ablation_changes_output':True,'latency':clean['provenance']['latency_seconds']});print(id,clean['predictions'][0],flush=True)
 (out/'gallery.json').write_text(json.dumps(gallery));(out/'projection.json').write_text(json.dumps({'calibration':CALIBRATION,'sha256':engine.basis_hash,'heads':engine.basis}));(root/'docs/validation.json').write_text(json.dumps({'parity_max_abs':engine.parity,'checks':checks},indent=2))

import {test} from 'node:test';import assert from 'node:assert/strict';import {alignTokens} from '../src/alignment.mjs';
const run=(prompt,pieces)=>({prompt,tokens:pieces.map(([text,start,id])=>({id,offset:[start,start+text.length]}))});
test('unchanged spans align after insertion',()=>{assert.deepEqual(alignTokens(run('cat dog',[['cat',0,1],[' dog',3,2]]),run('cat big dog',[['cat',0,1],[' big',3,3],[' dog',7,2]])),[0,null]);});
test('identical prompt preserves every token',()=>{const a=run(' red red',[[' red',0,1],[' red',4,1]]);assert.deepEqual(alignTokens(a,a),[0,1])});
test('ambiguous repeated word deletion remains unpaired',()=>{assert.deepEqual(alignTokens(run('cat cat',[['cat',0,1],[' cat',3,2]]),run('cat',[['cat',0,1]])),[null,null])});
test('token splits and different ids remain unpaired',()=>{assert.deepEqual(alignTokens(run('hello',[['hello',0,1]]),run('hello',[['he',0,2],['llo',2,3]])),[null])});
test('crossing alternatives are not falsely paired',()=>{const a={prompt:'ab',tokens:[{id:1,offset:[0,1]},{id:2,offset:[1,2]}]},b={prompt:'ba',tokens:[{id:2,offset:[0,1]},{id:1,offset:[1,2]}]};assert.deepEqual(alignTokens(a,b),[null,null])});

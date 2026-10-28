export type Token={id:number;text:string;offset:[number,number];boundary?:boolean};
export type Candidate={id:number;text:string;p:number};
export type Head={extent?:number;attention:number[][];q:number[][];k:number[][]};
export type Run={id:string;title?:string;prompt:string;tokens:Token[];predictions:Candidate[];concentration:number[][][];heads:Head[][];provenance:Record<string,unknown>;intervention?:{layer:number;head:number};baseline_id?:string;ablation?:string;probabilities?:number[]};
export type Gallery={id:string;title:string;prompt:string;file:string;ablation:string;head:[number,number]};

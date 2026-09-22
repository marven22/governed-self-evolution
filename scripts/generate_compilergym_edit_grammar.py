"""Generate the first bounded, reproducible compiler-policy edit grammar."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--action-space-size',type=int,default=124); p.add_argument('--seed',type=int,default=20260922); a=p.parse_args()
 r=random.Random(a.seed); x,y=r.sample(range(a.action_space_size),2)
 grammar={'protocol':'compilergym-edit-grammar-v1','seed':a.seed,'edits':[
  {'label':'HOLD','op':'hold'}, {'label':'TRUNCATE_2','op':'truncate','count':2},
  {'label':f'APPEND_{x}','op':'append','action':x},{'label':f'APPEND_{y}','op':'append','action':y},
  {'label':f'REPLACE_LAST_{x}','op':'replace_last','action':x},{'label':f'REPLACE_LAST_{y}','op':'replace_last','action':y}]}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(grammar,indent=2,sort_keys=True)+'\n'); print(json.dumps(grammar))
if __name__=='__main__': main()

"""MoA v3 Fine-Tuning — RunPod Serverless Training Handler"""
import os, sys, json, time, re, math, subprocess

def ensure_deps():
    missing = []
    for pkg, mod in [("scipy", "scipy"), ("numpy", "numpy"), ("datasets", "datasets")]:
        try: __import__(mod)
        except ImportError: missing.append(pkg)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing)
ensure_deps()

import numpy as np
from scipy.optimize import minimize
from datasets import load_dataset

CW = {"word_count":0.04,"question":0.02,"code":0.18,"imperative":0.12,"math":0.05,"multi_step":0.08,"constraints":0.06,"context":0.05,"sentence_count":0.03,"avg_word_length":0.02,"architecture":0.28,"technical_design":0.18,"four_plus":0.10}
WN = list(CW.keys())

def features(p):
    t=p.lower(); w=t.split(); wc=len(w); sig=0
    if'?'in p:sig+=1
    if any(k in t for k in["code","function","def ","class ","import ","``"]):sig+=1
    if any(k in t for k in["write","create","build","implement","generate","fix","debug","optimize"]):sig+=1
    if re.search(r'\d+[\s]*[+\-*/=]',p):sig+=1
    if any(k in t for k in["first","then","finally","step","part","section"]):sig+=1
    if any(k in t for k in["must","should","required","only","don't","cannot","limit"]):sig+=1
    if any(k in t for k in["given","consider","assume","suppose","based on","according to"]):sig+=1
    if any(k in t for k in["architecture","design pattern","microservice","monolith","mvc","rest api","system design","component"]):sig+=1
    if any(k in t for k in["schema","database","api endpoint","endpoint","authentication","authorization","deployment","pipeline"]):sig+=1
    sc=len(re.split(r'[.!?]+',p)); ri=len(set(w))/wc if wc>0 else 0
    return{"word_count":wc,"has_question":1 if"?"in p else 0,"has_code":1 if any(k in t for k in["code","function","def ","class ","import ","``"]) else 0,"has_imperative":1 if any(k in t for k in["write","create","build","implement","generate","fix","debug","optimize"]) else 0,"has_math":1 if re.search(r'\d+[\s]*[+\-*/=]',p) else 0,"has_multi_step":1 if any(k in t for k in["first","then","finally","step","part","section"]) else 0,"has_constraints":1 if any(k in t for k in["must","should","required","only","don't","cannot","limit"]) else 0,"has_context":1 if any(k in t for k in["given","consider","assume","suppose","based on","according to"]) else 0,"sentence_count":sc,"avg_word_length":np.mean([len(x)for x in w])if w else 0,"has_architecture":1 if any(k in t for k in["architecture","design pattern","microservice","monolith","mvc","rest api","system design","component"]) else 0,"has_technical_design":1 if any(k in t for k in["schema","database","api endpoint","endpoint","authentication","authorization","deployment","pipeline"]) else 0,"char_count":len(p),"unique_words":len(set(w)),"synthetic_complexity":min(0.1*math.log1p(wc)/math.log1p(200)+min(sig*0.1,0.9)+min(sc*0.02,0.1)+ri*0.05,1.0)}

def cscore(f,w):
    s=0.0
    s+=w.get("word_count",0.04)*math.log1p(f["word_count"])/6.0
    s+=w.get("question",0.02)*f["has_question"]+w.get("code",0.18)*f["has_code"]+w.get("imperative",0.12)*f["has_imperative"]+w.get("math",0.05)*f["has_math"]+w.get("multi_step",0.08)*f["has_multi_step"]+w.get("constraints",0.06)*f["has_constraints"]+w.get("context",0.05)*f["has_context"]
    s+=w.get("sentence_count",0.03)*min(f["sentence_count"],10)/10.0+w.get("avg_word_length",0.02)*f["avg_word_length"]/10.0+w.get("architecture",0.28)*f["has_architecture"]+w.get("technical_design",0.18)*f["has_technical_design"]
    if f["word_count"]<10 and f["has_architecture"]==0:s*=0.7
    fc=sum(v for v in f.values()if isinstance(v,(int,float))and v>0 and v not in(f["word_count"],f["char_count"],f["avg_word_length"],f["sentence_count"],f["unique_words"]))
    if fc>=4:s+=w.get("four_plus",0.10)
    return min(max(s,0.0),1.0)

def tier(s):
    if s<0.08:return"trivial"
    if s<0.18:return"light"
    if s<0.32:return"moderate"
    if s<0.52:return"heavy"
    if s<0.72:return"intensive"
    return"extreme"

def handler(event):
    inp=event.get("input",{})
    ds_names=inp.get("datasets",["alpaca"]); mp=inp.get("max_per",10000)
    catalogs={"alpaca":{"hf":"tatsu-lab/alpaca","k":"instruction"},"openorca":{"hf":"Open-Orca/OpenOrca","k":"question"}}
    prompts=[]
    for n in ds_names:
        if n not in catalogs:continue
        c=catalogs[n]
        try:
            d=load_dataset(c["hf"],split=f"train[:{mp}]")
            b=[x[c["k"]]for x in d if c["k"]in x and isinstance(x.get(c["k"]),str)]
            prompts.extend(b[:mp])
        except Exception as e:print(f"Failed {n}: {e}")
    if not prompts:return{"error":"No prompts loaded"}
    feats=[features(p)for p in prompts]; labs=[f["synthetic_complexity"]for f in feats]
    sp=int(len(feats)*0.8); tf,tl=feats[:sp],labs[:sp]; ef,el=feats[sp:],labs[sp:]
    def obj(wa):
        w=dict(zip(WN,wa)); return np.mean([(cscore(f,w)-t)**2 for f,t in zip(tf,tl)])
    r=minimize(obj,np.array([CW[n]for n in WN]),method="L-BFGS-B",bounds=[(0.0,0.5)]*len(WN),options={"maxiter":2000})
    opt=dict(zip(WN,r.x))
    ta=sum(1 for f,t in zip(ef,el)if tier(cscore(f,opt))==tier(t))/len(ef)if ef else 0
    ba=sum(1 for f,t in zip(ef,el)if tier(cscore(f,CW))==tier(t))/len(ef)if ef else 0
    return{"version":"v3.0","datasets":ds_names,"total":len(prompts),"train":len(tf),"test":len(ef),"optimized_weights":{k:round(v,4)for k,v in opt.items()},"baseline_accuracy":round(ba,4),"optimized_accuracy":round(ta,4),"improvement":round(ta-ba,4),"status":"completed"}

if __name__=="__main__":
    print(json.dumps(handler({"input":{"datasets":["alpaca"],"max_per":10000}}),indent=2))

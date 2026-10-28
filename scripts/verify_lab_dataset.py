#!/usr/bin/env python3
"""verify_lab_dataset.py — integrity checks and single-feature-group LOGO ablation on labeled.csv.
Produces the numbers of Tables 2.3 and 2.5 (Chapter 2).
Usage: python3 scripts/verify_lab_dataset.py data/processed/labeled.csv > reports/lab_verification.json
"""
import json, sys, importlib.util, numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
d = pd.read_csv(sys.argv[1]); out = {}
out["rows"]=len(d); out["dup_session_ids"]=int(d.session.duplicated().sum())
out["null_cells"]=int(d.isna().sum().sum()); out["null_by_col"]={k:int(v) for k,v in d.isna().sum().items() if v}
ts=pd.to_datetime(d.start_utc, utc=True); out["time_range"]=[str(ts.min()), str(ts.max())]
lab=d[d.labeled]; out["labeled"]=len(lab)
out["labeled_by_population"]=lab.groupby(["group","population"]).size().unstack(fill_value=0).to_dict()
u=d.login_username.astype("string").fillna("")
out["username_regex_consistency"]=bool(((u.str.match(r"^p\d+$")|u.str.match(r"^bot\d+$"))==d.labeled).all())
out["attack_tool_usernames_top"]=d[~d.labeled].login_username.fillna("<none>").value_counts().head(8).to_dict()
out["attack_tool_interactive"]=int(((~d.labeled)&(d.population=="interactive")).sum())
i=lab[lab.population=="interactive"].copy()
F=["command_count","unique_commands","command_diversity","icd_mean","icd_std","session_duration","has_tty","single_char_ratio","keystroke_gap_median","backspaces"]
for c in F: i[c]=pd.to_numeric(i[c],errors="coerce").fillna(0.0)
out["interactive"]={"n":len(i),"human":int(i.is_human.sum()),"bot":int((1-i.is_human).sum()),"groups":int(i.group.nunique())}
out["bot_command_count_values"]=sorted(i[i.is_human==0].command_count.unique().tolist())
out["human_command_count_range"]=[int(i[i.is_human==1].command_count.min()),int(i[i.is_human==1].command_count.max())]
out["has_tty_by_class"]=i.groupby("is_human").has_tty.mean().round(3).to_dict()
out["icd_mean_range"]={"human":[round(i[i.is_human==1].icd_mean.min(),2),round(i[i.is_human==1].icd_mean.max(),2)],
  "bot3":[round(i[i.group=="bot3"].icd_mean.min(),2),round(i[i.group=="bot3"].icd_mean.max(),2)]}
out["src_ip_by_class"]={str(k):sorted(v) for k,v in i.groupby("is_human").src_ip.unique().items()}
def logo(feats, mk):
    X,y,g=i[feats].values,i.is_human.values,i.group.values; yt,yp=[],[]
    for tr,te in LeaveOneGroupOut().split(X,y,g):
        clf=mk().fit(X[tr],y[tr]); yt+=list(y[te]); yp+=list(clf.predict(X[te]))
    return round(matthews_corrcoef(yt,yp),4)
mk={"DT":lambda: make_pipeline(StandardScaler(),DecisionTreeClassifier(class_weight="balanced",random_state=42)),
    "RF":lambda: make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42)),
    "LogReg":lambda: make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",max_iter=1000))}
sets={"command_count_only":["command_count"],
      "timing_without_counts":["icd_mean","icd_std","session_duration"],
      "keystroke_only":["single_char_ratio","keystroke_gap_median","backspaces"],
      "has_tty_only":["has_tty"]}
out["ablation_mcc"]={s:{m:logo(f,mk[m]) for m in mk} for s,f in sets.items()}
print(json.dumps(out,indent=1,default=str))

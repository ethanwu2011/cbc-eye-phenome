"""Figure 3: the eye-versus-CBC partition with a two-year separation between eye and outcome.

A) The index-anchored landscape: C-index with demographics plus eye against demographics plus
   CBC for the 449 screened phenotypes, every outcome at least 730 days after the index.
   Classes are effect-size strata (C-index difference of at least 0.030), as in the Methods.
B) The same PheCodes concurrently and index-anchored (identical codes, no mapping step).
C) Organ-system ordering, concurrent against index-anchored.
D) Agreement with the concurrent landscape at required leads of 90, 365 and 730 days.

Correlations carry bootstrap 95% CIs (1000 phenotype-level resamples); per-phenotype
classes carry 400-resample patient-level bootstrap CIs with BH-FDR.
"""
from config import CBCEYE_ROOT
import pandas as pd, numpy as np, plotly.graph_objects as go
from scipy import stats
HERE=CBCEYE_ROOT; OUT=f"{HERE}/final_manuscript/figures"
EYE="#2b6e6e"; CBCc="#9c5b5b"; GREY="#c9c9c9"; SYN="#2e7d52"
COL={"synergy":SYN,"eye-specific":EYE,"CBC-specific":CBCc,"comparable":GREY}
rng=np.random.default_rng(42)

xs=pd.read_parquet(f"{HERE}/landscape_full.parquet"); xs["phecode"]=xs.phecode.astype(str)
# Apply the SAME concurrent screen used for every other concurrent result. Without this the
# figure merges unscreened concurrent phenotypes (including tautological ones such as anemia)
# against screened prospective ones, which inflates n and mixes two different analysis sets.
xs=xs[(xs.circularity_tier=="clean")&(xs.klass!="no signal")]
xs["adv_xs"]=xs.auc_eye-xs.auc_cbc; xs=xs.drop_duplicates("phecode")

def load(tag):
    # canonical_index_screen.py applies the concurrent eligibility rule and circularity screen
    p=pd.read_csv(f"{HERE}/prospective_landscape{tag}_canonical.csv",dtype={"phecode":str}).drop_duplicates("phecode")
    p=p[p.canonical]
    p["adv_pr"]=p.c_eye-p.c_cbc; return p
def boot_rho(a,b,B=1000):
    n=len(a); out=np.empty(B)
    for i in range(B):
        ix=rng.integers(0,n,n); out[i]=stats.spearmanr(a.iloc[ix],b.iloc[ix]).statistic
    return np.percentile(out,[2.5,97.5])
def boot_diff(m,B=1000):
    """median prospective advantage: eye-led minus CBC-led, bootstrapped over phenotypes."""
    e=m[m.adv_xs>0.03].adv_pr.to_numpy(); c=m[m.adv_xs<-0.03].adv_pr.to_numpy()
    out=np.empty(B)
    for i in range(B):
        out[i]=np.median(rng.choice(e,len(e)))-np.median(rng.choice(c,len(c)))
    return np.median(out),np.percentile(out,[2.5,97.5])

CELLS=[("_B90","≥90 days",90),("","≥1 year",365),("_B730","≥2 years",730)]
stats_rows=[]
for tag,lab,days in CELLS:
    try: pr=load(tag)
    except FileNotFoundError: print(f"  (missing {tag}, skipping)"); continue
    m=xs[["phecode","adv_xs"]].merge(pr[["phecode","adv_pr"]],on="phecode")
    r=stats.spearmanr(m.adv_xs,m.adv_pr); lo,hi=boot_rho(m.adv_xs,m.adv_pr)
    d,(dlo,dhi)=boot_diff(m)
    stats_rows.append(dict(lead=lab,days=days,n=len(m),rho=r.statistic,p=r.pvalue,lo=lo,hi=hi,
                           diff=d,dlo=dlo,dhi=dhi))
    print(f"  {lab:<10} n={len(m):>3} rho={r.statistic:.3f} [{lo:.3f},{hi:.3f}] p={r.pvalue:.1e} | eye-led−CBC-led {d:+.3f} [{dlo:+.3f},{dhi:+.3f}]",flush=True)
S=pd.DataFrame(stats_rows); S.to_csv(f"{HERE}/fig3_stats.csv",index=False)
main=[s for s in stats_rows if s["days"]==730][0]
pr1=load("_B730")
m1=xs[["phecode","adv_xs"]].merge(pr1[["phecode","adv_pr","events","description"]],on="phecode")

# ---------------- A ----------------
# the screened set, 449 of 533, matching the Results and the Figure 3 legend.
sig=pd.read_csv(f"{HERE}/prospective_landscape_B730_canonical.csv",dtype={"phecode":str})
sig=sig[sig.canonical].copy()
sig["klass"]=sig.klass.replace({"comparable":"indeterminate"})
excl=pd.DataFrame(columns=sig.columns)   # already screened upstream
sig["cls"]=sig.klass.map({"eye-specific":"eye-specific","CBC-specific":"CBC-specific","synergy":"synergy","indeterminate":"indeterminate (underpowered)"})
CMAP={"indeterminate (underpowered)":"#dcdcdc","CBC-specific":CBCc,"eye-specific":EYE,"synergy":SYN}
figA=go.Figure(); lo_,hi_=0.35,0.92
figA.add_trace(go.Scatter(x=[lo_,hi_],y=[lo_,hi_],mode="lines",line=dict(color="#ddd",width=1,dash="dash"),showlegend=False,hoverinfo="skip"))
for k in ["indeterminate (underpowered)","CBC-specific","synergy","eye-specific"]:
    s_=sig[sig.cls==k]
    if not len(s_): continue
    figA.add_trace(go.Scatter(x=s_.c_cbc,y=s_.c_eye,mode="markers",name=f"{k} ({len(s_)})",
        marker=dict(size=5+np.sqrt(s_.events.clip(0,900))*0.30,color=CMAP[k],
                    opacity=0.45 if k.startswith("indet") else 0.85,line=dict(width=0.4,color="#fff")),
        text=s_.description,hovertemplate="%{text}<br>CBC %{x:.3f} · eye %{y:.3f}<extra></extra>"))
figA.add_trace(go.Scatter(x=excl.c_cbc,y=excl.c_eye,mode="markers",name=f"excluded: circular \u2014 {len(excl)}",
    marker=dict(size=6,color="rgba(0,0,0,0)",line=dict(width=1.1,color="#b9b9b9")),
    text=excl.description,hovertemplate="%{text}<br>excluded<extra></extra>"))
figA.add_annotation(x=0.855,y=0.60,text="<b>CBC-specific</b>",showarrow=False,font=dict(size=13,color=CBCc))
figA.add_annotation(x=0.44,y=0.875,text="<b>eye-specific</b>",showarrow=False,font=dict(size=13,color=EYE))
figA.update_layout(template="simple_white",font=dict(family="Helvetica",size=12),width=700,height=650,
    title=dict(text=f"<b>A   The prospective landscape</b>  <sub>{len(sig)} phenotypes, every event \u22652 yr ahead</sub>",x=0.02),
    xaxis=dict(title="C-index, demographics + CBC",range=[lo_,hi_]),
    yaxis=dict(title="C-index, demographics + eye",range=[lo_,hi_],scaleanchor="x",scaleratio=1),
    legend=dict(font=dict(size=9),y=0.015,x=0.985,xanchor="right",yanchor="bottom",
                bgcolor="rgba(255,255,255,0.88)",bordercolor="#e5e5e5",borderwidth=1))
figA.write_image(f"{OUT}/F3_A.png",scale=3)

# ---------------- B ----------------
col=[EYE if a>0.03 else (CBCc if a<-0.03 else GREY) for a in m1.adv_xs]
figB=go.Figure()
figB.add_hline(y=0,line=dict(color="#ddd",width=1)); figB.add_vline(x=0,line=dict(color="#ddd",width=1))
figB.add_trace(go.Scatter(x=m1.adv_xs,y=m1.adv_pr,mode="markers",showlegend=False,
    marker=dict(size=5+np.sqrt(m1.events.clip(0,900))*0.28,color=col,opacity=0.75,line=dict(width=0.4,color="#fff")),
    text=m1.description,hovertemplate="%{text}<br>concurrent %{x:.3f}<br>prospective %{y:.3f}<extra></extra>"))
z=np.polyfit(m1.adv_xs,m1.adv_pr,1); xr=np.linspace(m1.adv_xs.min(),m1.adv_xs.max(),50)
figB.add_trace(go.Scatter(x=xr,y=np.polyval(z,xr),mode="lines",line=dict(color="#333",width=1.6,dash="dash"),showlegend=False))
figB.add_annotation(x=0.06,y=0.122,xanchor="left",align="left",showarrow=False,font=dict(size=11.5,color="#333"),
    text=f"<b>ρ = {main['rho']:.2f}</b>  [{main['lo']:.2f}, {main['hi']:.2f}]<br>p = {main['p']:.0e} · n = {main['n']}")
figB.add_annotation(x=0.155,y=-0.115,text="eye-led concurrently<br>stays eye-led 2 yr ahead",showarrow=False,font=dict(size=10,color=EYE))
figB.update_layout(template="simple_white",font=dict(family="Helvetica",size=12),width=700,height=650,
    title=dict(text="<b>B   Same diseases, concurrent vs ≥2 yr ahead</b>  <sub>identical PheCodes</sub>",x=0.02),
    xaxis=dict(title="eye advantage over CBC, concurrent (AUROC)",range=[-0.30,0.22]),
    yaxis=dict(title="eye advantage over CBC, ≥2 yr ahead (C-index)",range=[-0.17,0.15]))
figB.write_image(f"{OUT}/F3_B.png",scale=3)

# ---------------- C ----------------
a=xs.groupby("group").adv_xs.median(); b=pr1.groupby("group").adv_pr.median()
o=pd.DataFrame({"concurrent":a,"prospective":b}).dropna()
o=o[[len(xs[xs.group==g])>=8 for g in o.index]].sort_values("concurrent")
ro=stats.spearmanr(o.concurrent,o.prospective)
figC=go.Figure()
for g,row in o.iterrows():
    figC.add_trace(go.Scatter(x=[row.concurrent,row.prospective],y=[g,g],mode="lines",line=dict(color="#e6e6e6",width=3),showlegend=False,hoverinfo="skip"))
figC.add_trace(go.Scatter(x=o.concurrent,y=o.index,mode="markers",name="concurrent",marker=dict(color="#bdbdbd",size=10,line=dict(width=0.5,color="#fff"))))
figC.add_trace(go.Scatter(x=o.prospective,y=o.index,mode="markers",name="≥2 yr ahead",marker=dict(color=EYE,size=12,line=dict(width=0.5,color="#fff"))))
figC.add_vline(x=0,line=dict(color="#bbb",width=1))
figC.add_annotation(x=-0.135,y=len(o)-1.3,xanchor="left",align="left",showarrow=False,font=dict(size=11,color="#333"),
    text=f"<b>ordering preserved</b><br>ρ = {ro.statistic:.2f} (p = {ro.pvalue:.3f})")
figC.update_layout(template="simple_white",font=dict(family="Helvetica",size=12),width=700,height=650,
    title=dict(text="<b>C   Organ-system ordering is preserved</b>",x=0.02),
    xaxis=dict(title="median eye advantage over CBC"),margin=dict(l=175),
    legend=dict(orientation="h",y=-0.10,x=0.5,xanchor="center"))
figC.write_image(f"{OUT}/F3_C.png",scale=3)

# ---------------- D: dose-response ----------------
figD=go.Figure()
figD.add_trace(go.Scatter(x=S.days,y=S.rho,mode="lines+markers",showlegend=False,
    line=dict(color=EYE,width=2.5),marker=dict(size=13,color=EYE,line=dict(width=1,color="#fff")),
    error_y=dict(type="data",symmetric=False,array=S.hi-S.rho,arrayminus=S.rho-S.lo,color=EYE,width=6,thickness=1.5)))
for _,r in S.iterrows():
    figD.add_annotation(x=r.days,y=r.hi+0.035,text=f"<b>ρ={r.rho:.2f}</b><br><span style='font-size:9px'>p={r.p:.0e}</span>",showarrow=False,font=dict(size=10.5,color="#333"))
figD.add_hline(y=0,line=dict(color="#bbb",width=1))
figD.update_layout(template="simple_white",font=dict(family="Helvetica",size=12),width=700,height=430,
    title=dict(text="<b>D   The partition attenuates with required lead, but persists</b>",x=0.02),
    xaxis=dict(title="required lead time before diagnosis (days)",tickvals=list(S.days),
               ticktext=[f"{d}<br><span style='font-size:9px'>n={n}</span>" for d,n in zip(S.days,S.n)],range=[0,820]),
    yaxis=dict(title="ρ  (concurrent vs index-anchored)",range=[0,0.55]))
figD.write_image(f"{OUT}/F3_D.png",scale=3)

from PIL import Image
A,Bi,C,D=[Image.open(f"{OUT}/F3_{k}.png") for k in "ABCD"]
W=A.width+Bi.width; H=A.height+max(C.height,D.height)
cv=Image.new("RGB",(W,H),(255,255,255))
cv.paste(A,(0,0)); cv.paste(Bi,(A.width,0)); cv.paste(C,(0,A.height)); cv.paste(D,(A.width,A.height+(C.height-D.height)//2))
cv.save(f"{OUT}/Figure3_prospective.png")
print("\nsaved Figure3_prospective.png",cv.size)
print("saved fig3_stats.csv")

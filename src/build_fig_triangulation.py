"""Figure — the partition reproduces across three designs.

The per-disease comparison uses identical PheCodes within MIMIC, so there is no mapping step
at all, and NHANES enters at the organ level where its 66 measured conditions are comparable.

A) Per-disease: the eye's advantage over the CBC concurrently (x) vs >=1 year ahead (y),
   565 shared phenotypes. If the partition were an artifact of concurrent coding, this
   would be a cloud.
B) Organ systems: the same ordering, concurrent vs prospective.
C) Taxonomy composition across all three designs.
"""
from config import CBCEYE_ROOT
import pandas as pd, numpy as np, plotly.graph_objects as go
from scipy import stats
HERE=CBCEYE_ROOT; OUT=f"{HERE}/final_manuscript/figures"
EYE="#2b6e6e"; CBC="#9c5b5b"; GREY="#c9c9c9"

t=pd.read_csv(f"{HERE}/triangulation.csv",dtype={"phecode":str})
xs=pd.read_parquet(f"{HERE}/landscape_full.parquet"); xs["phecode"]=xs.phecode.astype(str)
pr=pd.read_csv(f"{HERE}/prospective_landscape.csv",dtype={"phecode":str})
nh=pd.read_csv(f"{HERE}/nhanes_validation_results.csv")

# ---------- A: per-disease concurrent vs prospective ----------
r,p=stats.spearmanr(t.xs_eye_adv,t.pr_eye_adv)
col=[EYE if a>0.03 else (CBC if a<-0.03 else GREY) for a in t.xs_eye_adv]
figA=go.Figure()
figA.add_hline(y=0,line=dict(color="#ddd",width=1)); figA.add_vline(x=0,line=dict(color="#ddd",width=1))
figA.add_trace(go.Scatter(x=t.xs_eye_adv,y=t.pr_eye_adv,mode="markers",showlegend=False,
    marker=dict(size=5+np.sqrt(t.events.clip(0,900))*0.28,color=col,opacity=0.75,line=dict(width=0.4,color="#fff")),
    text=t.description,hovertemplate="%{text}<br>concurrent %{x:.3f}<br>prospective %{y:.3f}<extra></extra>"))
z=np.polyfit(t.xs_eye_adv,t.pr_eye_adv,1); xr=np.linspace(t.xs_eye_adv.min(),t.xs_eye_adv.max(),50)
figA.add_trace(go.Scatter(x=xr,y=np.polyval(z,xr),mode="lines",line=dict(color="#333",width=1.6,dash="dash"),showlegend=False))
figA.add_annotation(x=0.16,y=0.115,text=f"<b>Spearman ρ = {r:.2f}</b><br>p = {p:.1e} · n = {len(t)}",
    showarrow=False,font=dict(size=12,color="#333"),align="left")
figA.add_annotation(x=0.15,y=-0.10,text="eye-led concurrently<br>stays eye-led ahead",showarrow=False,font=dict(size=10,color=EYE))
figA.update_layout(template="simple_white",font=dict(family="Helvetica",size=12),width=680,height=620,
    title=dict(text="<b>A   Same diseases, concurrent vs ≥1 year ahead</b>  <sub>(MIMIC, identical PheCodes)</sub>",x=0.02),
    xaxis=dict(title="eye advantage over CBC — concurrent (AUROC)",range=[-0.30,0.22]),
    yaxis=dict(title="eye advantage over CBC — ≥1 yr ahead (C-index)",range=[-0.16,0.14]))
figA.write_image(f"{OUT}/Fig_tri_A.png",scale=3)

# ---------- B: organ ordering ----------
xs["adv"]=xs.auc_eye-xs.auc_cbc; pr["adv"]=pr.c_eye-pr.c_cbc
a=xs.groupby("group").adv.median(); b=pr.groupby("group").adv.median()
o=pd.DataFrame({"concurrent":a,"prospective":b}).dropna()
o=o[[len(xs[xs.group==g])>=8 for g in o.index]].sort_values("concurrent")
ro,po=stats.spearmanr(o.concurrent,o.prospective)
figB=go.Figure()
for g,row in o.iterrows():
    figB.add_trace(go.Scatter(x=[row.concurrent,row.prospective],y=[g,g],mode="lines",
        line=dict(color="#e6e6e6",width=3),showlegend=False,hoverinfo="skip"))
figB.add_trace(go.Scatter(x=o.concurrent,y=o.index,mode="markers",name="concurrent",
    marker=dict(color="#bdbdbd",size=10,line=dict(width=0.5,color="#fff"))))
figB.add_trace(go.Scatter(x=o.prospective,y=o.index,mode="markers",name="≥1 yr ahead",
    marker=dict(color=EYE,size=12,line=dict(width=0.5,color="#fff"))))
figB.add_vline(x=0,line=dict(color="#bbb",width=1))
figB.add_annotation(x=-0.13,y=len(o)-1.4,text=f"<b>organ ordering preserved</b><br>Spearman ρ = {ro:.2f} (p = {po:.3f})",
    showarrow=False,font=dict(size=11,color="#333"),align="left")
figB.update_layout(template="simple_white",font=dict(family="Helvetica",size=12),width=680,height=620,
    title=dict(text="<b>B   Organ-system ordering is preserved in time</b>",x=0.02),
    xaxis=dict(title="median eye advantage over CBC"),margin=dict(l=170),
    legend=dict(orientation="h",y=-0.11,x=0.5,xanchor="center"))
figB.write_image(f"{OUT}/Fig_tri_B.png",scale=3)

# ---------- C: taxonomy across three designs ----------
xs["klass"]=np.where(xs.auc_eye>xs.auc_cbc+0.03,"eye-specific",
            np.where(xs.auc_cbc>xs.auc_eye+0.03,"CBC-specific","comparable"))
def frac(s):
    v=s.value_counts(normalize=True)*100
    return [v.get(k,0) for k in ["eye-specific","synergy","comparable","CBC-specific"]]
designs=[("MIMIC concurrent",frac(xs.klass)),("MIMIC ≥1 yr ahead",frac(pr.klass)),("NHANES imaging",frac(nh.klass))]
cls=["eye-specific","synergy","comparable","CBC-specific"]; cols=[EYE,"#2e7d52",GREY,CBC]
figC=go.Figure()
for i,(cl,c) in enumerate(zip(cls,cols)):
    figC.add_trace(go.Bar(name=cl,y=[d[0] for d in designs],x=[d[1][i] for d in designs],orientation="h",
        marker=dict(color=c),text=[f"{d[1][i]:.0f}%" if d[1][i]>=5 else "" for d in designs],
        textposition="inside",insidetextanchor="middle"))
figC.update_layout(template="simple_white",font=dict(family="Helvetica",size=12),width=680,height=380,barmode="stack",
    title=dict(text="<b>C   The same composition in all three designs</b>",x=0.02),
    xaxis=dict(title="% of phenotypes",range=[0,100]),margin=dict(l=150),
    legend=dict(orientation="h",y=-0.22,x=0.5,xanchor="center"))
figC.write_image(f"{OUT}/Fig_tri_C.png",scale=3)

from PIL import Image
A,Bi,C=[Image.open(f"{OUT}/Fig_tri_{p}.png") for p in "ABC"]
W=A.width+Bi.width; H=max(A.height,Bi.height)+C.height
cv=Image.new("RGB",(W,H),(255,255,255))
cv.paste(A,(0,0)); cv.paste(Bi,(A.width,0)); cv.paste(C,((W-C.width)//2,max(A.height,Bi.height)))
cv.save(f"{OUT}/Fig_triangulation.png")
print("saved Fig_triangulation.png",cv.size)
print(f"  A: per-disease rho={r:.3f} p={p:.2g} n={len(t)}")
print(f"  B: organ rho={ro:.3f} p={po:.4f} over {len(o)} systems")

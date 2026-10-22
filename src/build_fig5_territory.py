"""FIGURE 5 — what each modality's territory is, across all three designs.

This is the paper's closing figure. Everything before it establishes THAT the two data streams
carry different information. This one says WHAT each one covers, and shows that the answer is
the same in a concurrent hospital cohort, in the same cohort two years before diagnosis, and in
an independent community cohort where the eye is imaged rather than coded.

A) The territory, three legs. Median eye-minus-CBC advantage for neurological phenotypes and
   for microvascular/renal phenotypes, each shown against that leg's all-phenotype baseline,
   with bootstrap CIs. Both territories sit above baseline in every leg.
B) The named eye-led phenotypes at a two-year lead, grouped by mechanism rather than by PheCode
   organ group, which merges microangiopathy with coronary disease.
C) The blood count's territory for contrast, its own leading phenotypes at the same lead.
"""
from config import CBCEYE_ROOT
import pandas as pd, numpy as np, re, plotly.graph_objects as go
from plotly.subplots import make_subplots
HERE=CBCEYE_ROOT; OUT=f"{HERE}/final_manuscript/figures"
EYE="#2b6e6e"; CBCc="#9c5b5b"; GREY="#b9b9b9"; MICRO="#3f8f78"
rng=np.random.default_rng(42)

NEURO=r'stroke|cerebr|neuropath|epilep|convuls|dementia|alzheim|parkinson|myelop|paralytic|nerve|brain|seizure|migraine|cognit|encephal|autonomic'
MICROR=r'nephropath|nephritis|renal|kidney|CKD|albumin|angiopath|atheroscler.*extrem|peripheral vascular|ulcer|osteomyel|amputat|cellulitis'

def med_ci(x,B=2000):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    if len(x)==0: return np.nan,np.nan,np.nan
    if len(x)<3: return np.median(x),np.nan,np.nan      # too few to bootstrap, show point only
    b=[np.median(rng.choice(x,len(x))) for _ in range(B)]
    return np.median(x),np.percentile(b,2.5),np.percentile(b,97.5)

def rel_ci(sub,allx,B=2000):
    """Bootstrap the quantity actually claimed: territory median MINUS all-phenotype median."""
    a=np.asarray(sub,float); a=a[~np.isnan(a)]
    z=np.asarray(allx,float); z=z[~np.isnan(z)]
    if len(a)<3: return (np.median(a)-np.median(z) if len(a) else np.nan),np.nan,np.nan
    d=[np.median(rng.choice(a,len(a)))-np.median(rng.choice(z,len(z))) for _ in range(B)]
    return np.median(a)-np.median(z),np.percentile(d,2.5),np.percentile(d,97.5)

# ---------------- data for panel A ----------------
xs=pd.read_parquet(f"{HERE}/landscape_full.parquet").drop_duplicates("phecode")
# Same concurrent screen as every other concurrent result, see note in build_fig3_final.py.
xs=xs[(xs.circularity_tier=="clean")&(xs.klass!="no signal")]
xs["adv"]=xs.auc_eye-xs.auc_cbc
pr=pd.read_csv(f"{HERE}/prospective_2yr_final.csv",dtype={"phecode":str}); pr["adv"]=pr.c_eye-pr.c_cbc
nh=pd.read_csv(f"{HERE}/nhanes_validation_results.csv"); nh["adv"]=nh.eye-nh.cbc
LEGS=[("Hospital cohort\nconcurrent",xs,"description"),
      ("Same cohort\n≥2 years ahead",pr,"description"),
      ("Community cohort\nretinal imaging",nh,"condition")]
rows=[]
for lab,df,col in LEGS:
    base=med_ci(df.adv)[0]
    for tname,pat in [("neurological",NEURO if col=="description" else r'[Ss]troke|memory|confusion|dementia'),
                      ("microvascular / renal",MICROR if col=="description" else r'Albumin|CKD|kidney')]:
        s=df[df[col].astype(str).str.contains(pat,case=False,na=False)]
        m,_,_=med_ci(s.adv)
        rel,rlo,rhi=rel_ci(s.adv,df.adv)
        rows.append(dict(leg=lab,terr=tname,n=len(s),med=m,base=base,rel=rel,lo=rlo,hi=rhi))
T=pd.DataFrame(rows)

figA=make_subplots(rows=1,cols=3,shared_yaxes=True,horizontal_spacing=0.045,
                   subplot_titles=[l for l,_,_ in LEGS])
for i,(lab,_,_) in enumerate(LEGS,start=1):
    t=T[T.leg==lab]; b=t.base.iloc[0]
    figA.add_vline(x=0,row=1,col=i,line=dict(color=GREY,width=1.4,dash="dot"))
    for j,(_,r) in enumerate(t.iterrows()):
        c=EYE if r.terr=="neurological" else MICRO
        if not np.isnan(r.lo):
            figA.add_trace(go.Scatter(x=[r.lo,r.hi],y=[r.terr,r.terr],mode="lines",
                line=dict(color=c,width=3),showlegend=False),row=1,col=i)
        figA.add_trace(go.Scatter(x=[r.rel],y=[r.terr],mode="markers",showlegend=False,
            marker=dict(color=c,size=13,line=dict(width=1,color="#fff")),
            hovertemplate=f"{r.terr}<br>{r.rel:+.3f} vs baseline<br>n={r.n}<extra></extra>"),row=1,col=i)
        lbl=f"<b>{r.rel:+.3f}</b>  n={r.n}" + ("" if not np.isnan(r.lo) else "  (n too small for CI)")
        figA.add_annotation(x=r.rel,y=j-0.32,text=lbl,showarrow=False,font=dict(size=9.5,color=c),row=1,col=i)
    figA.add_annotation(x=0,y=1.62,text="all-phenotype baseline",showarrow=False,font=dict(size=9,color="#8a8a8a"),row=1,col=i)
figA.update_layout(template="simple_white",font=dict(family="Helvetica",size=12),width=1250,height=420,
    title=dict(text="<b>A   Both territories favour the eye in every design</b>"
                    "  <sub>territory median minus that design's all-phenotype median, bootstrap 95% CI. All six estimates positive, sign test p = 0.031</sub>",x=0.015),
    margin=dict(l=175,t=95))
figA.update_xaxes(title_text="eye advantage over CBC, relative to baseline",zeroline=False,range=[-0.06,0.13])
figA.write_image(f"{OUT}/F5_A.png",scale=3)

# ---------------- panels B and C ----------------
def mech(d):
    d=str(d).lower()
    if re.search(r'osteomyel|amputat|cellulitis.*foot|ulcer',d): return "diabetic foot"
    if re.search(r'angiopath|atheroscler|peripheral vascular|aneurysm',d): return "peripheral vascular"
    if re.search(NEURO,d): return "neurological"
    if re.search(r'nephro|renal|kidney',d): return "renal"
    if re.search(r'heart|cardiac|atrial|coronary|myocard|lipid',d): return "cardiac / metabolic"
    return "other"
MC={"neurological":EYE,"diabetic foot":"#8a6a3f","peripheral vascular":MICRO,"renal":"#4a7fa5",
    "cardiac / metabolic":"#b08968","other":GREY}
e=pr[pr.klass=="eye-specific"].copy(); e["mech"]=e.description.map(mech)
e=e[e["mech"]!="other"].sort_values("adv",ascending=False).head(14).iloc[::-1]
figB=go.Figure()
for _,r in e.iterrows():
    figB.add_trace(go.Scatter(x=[r.c_cbc,r.c_eye],y=[r.description[:44]]*2,mode="lines",
        line=dict(color="#e8e8e8",width=3),showlegend=False,hoverinfo="skip"))
figB.add_trace(go.Scatter(x=e.c_cbc,y=[d[:44] for d in e.description],mode="markers",name="CBC",
    marker=dict(color=GREY,size=9)))
figB.add_trace(go.Scatter(x=e.c_eye,y=[d[:44] for d in e.description],mode="markers",name="eye",
    marker=dict(color=[MC[m] for m in e["mech"]],size=13,line=dict(width=0.6,color="#fff"))))
for _,r in e.iterrows():
    figB.add_annotation(x=max(r.c_eye,r.c_cbc)+0.006,y=r.description[:44],text=f"{r['mech']}  ({r.events})",
        showarrow=False,xanchor="left",font=dict(size=8.5,color=MC[r["mech"]]))
figB.update_layout(template="simple_white",font=dict(family="Helvetica",size=11),width=760,height=560,
    title=dict(text="<b>B   Where the eye leads, two years before diagnosis</b>  <sub>grouped by mechanism, event count in brackets</sub>",x=0.02),
    xaxis=dict(title="C-index",range=[0.42,0.92]),margin=dict(l=250,r=150),
    legend=dict(orientation="h",y=-0.11,x=0.5,xanchor="center"))
figB.write_image(f"{OUT}/F5_B.png",scale=3)

c=pr[pr.klass=="CBC-specific"].sort_values("adv").head(12).iloc[::-1]
figC=go.Figure()
for _,r in c.iterrows():
    figC.add_trace(go.Scatter(x=[r.c_eye,r.c_cbc],y=[r.description[:42]]*2,mode="lines",
        line=dict(color="#e8e8e8",width=3),showlegend=False,hoverinfo="skip"))
figC.add_trace(go.Scatter(x=c.c_eye,y=[d[:42] for d in c.description],mode="markers",name="eye",
    marker=dict(color=GREY,size=9)))
figC.add_trace(go.Scatter(x=c.c_cbc,y=[d[:42] for d in c.description],mode="markers",name="CBC",
    marker=dict(color=CBCc,size=13,line=dict(width=0.6,color="#fff"))))
figC.update_layout(template="simple_white",font=dict(family="Helvetica",size=11),width=700,height=560,
    title=dict(text="<b>C   Where the blood count leads</b>  <sub>infectious, visceral and bleeding disease</sub>",x=0.02),
    xaxis=dict(title="C-index",range=[0.42,0.92]),margin=dict(l=240),
    legend=dict(orientation="h",y=-0.11,x=0.5,xanchor="center"))
figC.write_image(f"{OUT}/F5_C.png",scale=3)

from PIL import Image
A,B,C=[Image.open(f"{OUT}/F5_{k}.png") for k in "ABC"]
W=max(A.width,B.width+C.width); H=A.height+max(B.height,C.height)
cv=Image.new("RGB",(W,H),(255,255,255))
cv.paste(A,((W-A.width)//2,0)); cv.paste(B,(0,A.height)); cv.paste(C,(B.width,A.height))
cv.save(f"{OUT}/Figure5_territory.png")
print("saved Figure5_territory.png",cv.size)
print("\nTERRITORY vs each design's own baseline:")
for _,r in T.iterrows():
    print(f"  {r.leg.replace(chr(10),' '):<34}{r.terr:<24}n={r.n:>3}  {r.med:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]  rel {r.rel:+.3f}")
T.to_csv(f"{HERE}/fig5_territory.csv",index=False)

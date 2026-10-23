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

# Mechanism territories, keyword-defined from the phenotype description.
# EXCLUDE removes lexical matches that are not mechanistic members: "myelop" caught
# myeloproliferative disease and spondylosis WITHOUT myelopathy, "paralytic" caught
# paralytic ileus, "ulcer" caught the gastrointestinal ulcers, and plain "renal"
# caught adrenal. Poisonings and neoplasms are dropped from both territories.
NEURO=r'stroke|cerebr|neuropath|epilep|convuls|dementia|alzheim|parkinson|myelopath|paralytic|\bnerve|\bbrain|seizure|migraine|cognit|encephal|autonomic'
MICROR=r'nephropath|nephritis|\brenal\b|kidney|CKD|albumin|angiopath|atheroscler.*extrem|peripheral vascular|ulcer|osteomyel|amputat|cellulitis'
EXCLUDE=(r'^Poisoning|adverse effect|without myelopath|myeloprolifer|paralytic ileus|'
         r'gastric ulcer|duodenal ulcer|peptic ulcer|gastrointestinal ulcer|ulcerative colitis|'
         r'cancer of|malignant neoplasm|benign neoplasm|secondary malignancy')

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
pr=pd.read_csv(f"{HERE}/prospective_landscape_B730_canonical.csv",dtype={"phecode":str})
pr=pr[pr.canonical].copy(); pr["adv"]=pr.c_eye-pr.c_cbc
nh=pd.read_csv(f"{HERE}/nhanes_validation_results.csv"); nh["adv"]=nh.eye-nh.cbc
LEGS=[("Hospital cohort\nconcurrent",xs,"description"),
      ("Same cohort\n≥2 years ahead",pr,"description"),
      ("Community cohort\nretinal imaging",nh,"condition")]
rows=[]
for lab,df,col in LEGS:
    base=med_ci(df.adv)[0]
    for tname,pat in [("neurological",NEURO if col=="description" else r'[Ss]troke|memory|confusion|dementia'),
                      ("microvascular / renal",MICROR if col=="description" else r'Albumin|CKD|kidney')]:
        _d=df[col].astype(str)
        s=df[_d.str.contains(pat,case=False,na=False,regex=True)
             & ~_d.str.contains(EXCLUDE,case=False,na=False,regex=True)]
        m,_,_=med_ci(s.adv)
        rel,rlo,rhi=rel_ci(s.adv,df.adv)
        rows.append(dict(leg=lab,terr=tname,n=len(s),med=m,base=base,rel=rel,lo=rlo,hi=rhi))
T=pd.DataFrame(rows)
T.to_csv(f"{HERE}/fig5_territory.csv",index=False)   # feeds the text and Supplementary S16

# ---------------- panels a, b, c : one per design ----------------
PANEL_FILES=[]
for i,(lab,_,_) in enumerate(LEGS):
    t=T[T.leg==lab]
    f=go.Figure()
    f.add_vline(x=0,line=dict(color=GREY,width=1.4,dash="dot"))
    for j,(_,r) in enumerate(t.iterrows()):
        c=EYE if r.terr=="neurological" else MICRO
        if not np.isnan(r.lo):
            f.add_trace(go.Scatter(x=[r.lo,r.hi],y=[r.terr,r.terr],mode="lines",
                line=dict(color=c,width=3),showlegend=False))
        f.add_trace(go.Scatter(x=[r.rel],y=[r.terr],mode="markers",showlegend=False,
            marker=dict(color=c,size=13,line=dict(width=1,color="#fff"))))
        lbl=f"<b>{r.rel:+.3f}</b>  n={r.n}" + ("" if not np.isnan(r.lo) else "  (n=2, no CI)")
        f.add_annotation(x=r.rel,y=j-0.30,text=lbl,showarrow=False,font=dict(size=11,color=c))
    f.add_annotation(x=0,y=1.58,text="all-phenotype baseline",showarrow=False,
                     font=dict(size=10,color="#8a8a8a"))
    f.update_layout(template="simple_white",font=dict(family="Helvetica",size=13),
        width=620,height=430,margin=dict(l=190,r=40,t=56,b=70),
        annotations=list(f.layout.annotations)+[dict(x=0.5,y=1.14,xref="paper",yref="paper",
            text=lab.replace(chr(10)," "),showarrow=False,font=dict(size=14),xanchor="center")])
    f.update_xaxes(title_text="eye advantage over CBC, relative to baseline",
                   zeroline=False,range=[-0.06,0.13])
    f.update_yaxes(categoryorder="array",categoryarray=["neurological","microvascular / renal"])
    fn=f"{OUT}/F5_{'abc'[i]}.png"; f.write_image(fn,scale=3); PANEL_FILES.append(fn)

# ---------------- panels B and C ----------------
def mech(d):
    d=str(d).lower()
    if re.search(EXCLUDE,d,re.I): return "other"   # dropped below with the other non-members
    if re.search(r'osteomyel|amputat|cellulitis.*foot|ulcer',d): return "diabetic foot"
    if re.search(r'angiopath|atheroscler|peripheral vascular|aneurysm',d): return "peripheral vascular"
    if re.search(NEURO,d,re.I): return "neurological"
    if re.search(r'nephro|renal|kidney',d): return "renal"
    if re.search(r'heart|cardiac|atrial|coronary|myocard|lipid',d): return "cardiac / metabolic"
    return "other"
MC={"neurological":EYE,"diabetic foot":"#8a6a3f","peripheral vascular":MICRO,"renal":"#4a7fa5",
    "cardiac / metabolic":"#b08968","other":GREY}
e=pr[pr.klass=="eye-specific"].copy(); e["mech"]=e.description.map(mech)
e=e[e["mech"]!="other"].sort_values("adv",ascending=False).head(14).iloc[::-1]
def lab(s):
    """Display label: drop the bracketed PheCode gloss and spell out EF."""
    s = re.sub(r"\s*\[[^\]]*\]", "", str(s)).strip()
    return s.replace(" EF", " ejection fraction")

figB=go.Figure()
for _,r in e.iterrows():
    figB.add_trace(go.Scatter(x=[r.c_cbc,r.c_eye],y=[lab(r.description)]*2,mode="lines",
        line=dict(color="#e8e8e8",width=3),showlegend=False,hoverinfo="skip"))
figB.add_trace(go.Scatter(x=e.c_cbc,y=[lab(d) for d in e.description],mode="markers",name="CBC",
    marker=dict(color=GREY,size=9)))
figB.add_trace(go.Scatter(x=e.c_eye,y=[lab(d) for d in e.description],mode="markers",name="eye",showlegend=False,
    marker=dict(color=[MC[m] for m in e["mech"]],size=13,line=dict(width=0.6,color="#fff"))))
figB.add_trace(go.Scatter(x=[None],y=[None],mode="markers",name="eye (colored by mechanism)",
    marker=dict(color="#555555",size=13)))
for _,r in e.iterrows():
    figB.add_annotation(x=max(r.c_eye,r.c_cbc)+0.006,y=lab(r.description),text=f"{r['mech']}  ({r.events})",
        showarrow=False,xanchor="left",font=dict(size=8.5,color=MC[r["mech"]]))
figB.update_layout(template="simple_white",font=dict(family="Helvetica",size=11),width=900,height=560,
    xaxis=dict(title="C-index",range=[0.42,0.92]),margin=dict(l=330,r=160),
    legend=dict(orientation="h",y=-0.11,x=0.5,xanchor="center"))
figB.write_image(f"{OUT}/F5_d.png",scale=3)

c=pr[pr.klass=="CBC-specific"].sort_values("adv").head(12).iloc[::-1]
figC=go.Figure()
for _,r in c.iterrows():
    figC.add_trace(go.Scatter(x=[r.c_eye,r.c_cbc],y=[lab(r.description)]*2,mode="lines",
        line=dict(color="#e8e8e8",width=3),showlegend=False,hoverinfo="skip"))
figC.add_trace(go.Scatter(x=c.c_eye,y=[lab(d) for d in c.description],mode="markers",name="eye",
    marker=dict(color=GREY,size=9)))
figC.add_trace(go.Scatter(x=c.c_cbc,y=[lab(d) for d in c.description],mode="markers",name="CBC",
    marker=dict(color=CBCc,size=13,line=dict(width=0.6,color="#fff"))))
figC.update_layout(template="simple_white",font=dict(family="Helvetica",size=11),width=880,height=560,
    xaxis=dict(title="C-index",range=[0.42,0.92]),margin=dict(l=330,r=60),
    legend=dict(orientation="h",y=-0.11,x=0.5,xanchor="center"))
figC.write_image(f"{OUT}/F5_e.png",scale=3)

from PIL import Image, ImageDraw, ImageFont
import os as _os
def _f(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf","/System/Library/Fonts/SFNS.ttf"):
        if _os.path.exists(p):
            try: return ImageFont.truetype(p,sz)
            except Exception: pass
    return ImageFont.load_default()
def _trim(im,th=735,pad=12):
    a=np.asarray(im.convert("RGB")).astype(int); ink=a.sum(axis=2)<th
    r=np.where(ink.any(axis=1))[0]; c=np.where(ink.any(axis=0))[0]
    return im.crop((max(0,c.min()-pad),max(0,r.min()-pad),
                    min(im.width,c.max()+pad),min(im.height,r.max()+pad)))

P={k:_trim(Image.open(f"{OUT}/F5_{k}.png")) for k in "abcde"}
MG,GU,LAB,LS=70,80,78,86
WIDTH=4000
top=[P[k] for k in "abc"]; bot=[P[k] for k in "de"]
ar=lambda ims:[i.width/i.height for i in ims]
h1=(WIDTH-2*MG-GU*2)/sum(ar(top))
h2=(WIDTH-2*MG-GU)/sum(ar(bot))
H=MG+LAB+int(h1)+LAB+int(h2)+MG+40
cv=Image.new("RGB",(WIDTH,H),"white"); dr=ImageDraw.Draw(cv); fnt=_f(LS)
y=MG+LAB; x=MG
for k,im in zip("abc",top):
    w=int(ar([im])[0]*h1); cv.paste(im.resize((w,int(h1)),Image.LANCZOS),(x,y))
    dr.text((x,y-LAB+4),k.upper()+".",fill=(26,28,32),font=fnt); x+=w+GU
y2=y+int(h1)+LAB+40; x=MG
for k,im in zip("de",bot):
    w=int(ar([im])[0]*h2); cv.paste(im.resize((w,int(h2)),Image.LANCZOS),(x,y2))
    dr.text((x,y2-LAB+4),k.upper()+".",fill=(26,28,32),font=fnt); x+=w+GU
cv.save(f"{OUT}/Figure5_territory.png")
print("saved Figure5_territory.png",cv.size,"(5 panels A-E, no panel titles)")

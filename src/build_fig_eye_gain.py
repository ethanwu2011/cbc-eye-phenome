"""Creative figure: 'What the eye adds, on top of the blood count.'
Gain-arrow (dumbbell) plot. For exemplar diseases, a maroon dot = AUROC with the CBC alone,
a colored arrow extends to AUROC with eye+CBC; arrow length = the eye's marginal contribution.
Arrow color: teal = eye-specific (eye carries it), green = synergy (eye complements the CBC).
Each arrow annotated with the dominant ophthalmic sign that drives it."""
from config import CBCEYE_ROOT
import pandas as pd, numpy as np, plotly.graph_objects as go
HERE=CBCEYE_ROOT; OUT=f"{HERE}/final_manuscript/figures"
EYEC,SYN,CBCC,GREYC="#2b6e6e","#2e7d52","#9c5b5b","#9a9a9a"
L=pd.read_parquet(f"{HERE}/landscape_full.parquet")
drv=pd.read_csv(f"{HERE}/modality_drivers.csv")[["name","top_eye_drivers"]]
d=L[(L.circularity_tier=="clean")&(L.klass!="no signal")].merge(drv,left_on="description",right_on="name",how="left")
d["gain"]=d.auc_both-d.auc_cbc

# curated, recognisable, mechanistically clean exemplars (matched by keyword)
PICK=["Arthropathy associated with neurological","Nephritis and nephropathy in diseases",
 "Peripheral autonomic neuropathy","Insulin pump user","Erectile dysfunction","Gastroparesis",
 "Cerebral artery occlusion, with cerebral infarction","Late effects of cerebrovascular",
 "intermittent claudication","Chronic ulcer of leg or foot","ulceration or gangrene",
 "Facial weakness","Unstable angina","Intracerebral hemorrhage","Lack of coordination","Hemiplegia",
 "Acute osteomyelitis"]
SHORT={"Arthropathy associated with neurological":"Charcot arthropathy (neuro)",
 "Nephritis and nephropathy in diseases":"Diabetic nephropathy","Peripheral autonomic neuropathy":"Peripheral autonomic neuropathy",
 "Insulin pump user":"Insulin-pump–treated diabetes","Erectile dysfunction":"Erectile dysfunction",
 "Gastroparesis":"Gastroparesis","Cerebral artery occlusion, with cerebral infarction":"Ischemic stroke",
 "Late effects of cerebrovascular":"Late effects of stroke","intermittent claudication":"Peripheral artery disease",
 "Chronic ulcer of leg or foot":"Chronic leg / foot ulcer","ulceration or gangrene":"Limb ischemia (gangrene)",
 "Facial weakness":"Facial weakness","Unstable angina":"Unstable angina","Intracerebral hemorrhage":"Intracerebral hemorrhage",
 "Lack of coordination":"Ataxia / incoordination","Hemiplegia":"Hemiplegia","Acute osteomyelitis":"Acute osteomyelitis"}
rows=[]
for key in PICK:
    sub=d[d.description.str.contains(key,case=False,regex=False)]
    if len(sub): rows.append(sub.sort_values("gain",ascending=False).iloc[0])
g=pd.DataFrame(rows).drop_duplicates("phecode")
g["label"]=[SHORT[k] for k in PICK if any(d.description.str.contains(k,case=False,regex=False))][:len(g)]

# clean the dominant eye-sign driver (strip weight; de-truncate common retinal terms)
CLEAN={"Background diabetic re":"background diabetic retinopathy","Proliferative diabetic":"proliferative retinopathy",
 "Dry eye syndrome of un":"dry-eye / ocular surface","Dry eye syndrome of bi":"dry-eye / ocular surface",
 "Unspecified macular de":"macular degeneration","Conjunctivitis, unspec":"conjunctivitis","Unspecified conjunctiv":"conjunctivitis",
 "Visual field defect":"visual-field defect","Diplopia":"diplopia","Nystagmus":"nystagmus","Ptosis":"ptosis",
 "homonymous":"homonymous field defect","heteronymous":"heteronymous field defect",
 "Retinal vascular occl":"retinal vascular occlusion","Optic":"optic-nerve sign"}
def topsign(s):
    if not isinstance(s,str) or not s.strip(): return ""
    first=s.split(";")[0].split("(")[0].strip()
    for k,v in CLEAN.items():
        if first.lower().startswith(k.lower()): return v
    return first.lower()
g["sign"]=g.top_eye_drivers.map(topsign)
g=g.sort_values("gain").reset_index(drop=True)   # smallest at bottom, biggest at top
print(g[["label","klass","auc_cbc","auc_both","gain","sign"]].round(3).to_string(index=False))

fig=go.Figure()
for i,r in g.iterrows():
    col={"eye-specific":EYEC,"synergy":SYN}.get(r.klass,GREYC)
    fig.add_trace(go.Scatter(x=[r.auc_cbc,r.auc_both],y=[i,i],mode="lines",
        line=dict(color=col,width=3),hoverinfo="skip",showlegend=False))
    fig.add_trace(go.Scatter(x=[r.auc_both],y=[i],mode="markers",
        marker=dict(symbol="triangle-right",size=12,color=col,line=dict(width=0.5,color="#fff")),
        hovertemplate=f"<b>{r.label}</b><br>CBC %{{x:.3f}}<extra></extra>",showlegend=False))
    fig.add_trace(go.Scatter(x=[r.auc_cbc],y=[i],mode="markers",
        marker=dict(symbol="circle",size=9,color=CBCC,line=dict(width=0.5,color="#fff")),
        hovertemplate=f"blood count alone: %{{x:.3f}}<extra></extra>",showlegend=False))
    txt=f"  <b>+{r.gain:.2f}</b>"+(f"  <i>{r.sign}</i>" if r.sign else "")
    fig.add_annotation(x=r.auc_both,y=i,text=txt,showarrow=False,xanchor="left",font=dict(size=9.5,color="#555"))
# legend proxies
for nm,c in [("blood count alone (CBC)",CBCC)]:
    fig.add_trace(go.Scatter(x=[None],y=[None],mode="markers",marker=dict(symbol="circle",size=9,color=c),name=nm))
for nm,c in [("+ eye, carries it (eye-specific)",EYEC),("+ eye, complements CBC (synergy)",SYN),
              ("+ eye (indeterminate)",GREYC)]:
    fig.add_trace(go.Scatter(x=[None],y=[None],mode="lines+markers",line=dict(color=c,width=3),
        marker=dict(symbol="triangle-right",size=11,color=c),name=nm))

fig.update_layout(template="simple_white",font=dict(family="Helvetica",size=13),
    title=dict(text="<b>What the eye adds, on top of the blood count</b>"
        "<br><sup>arrow = gain in AUROC when ophthalmic findings are added to the CBC · across all 528 diseases the median gain is +0.006, "
        "but a neuro-vascular tail gains up to +0.20</sup>",x=0.5),
    width=940,height=780,margin=dict(b=130),
    xaxis=dict(title="AUROC for systemic disease",range=[0.46,0.96],showgrid=True,gridcolor="#eee"),
    yaxis=dict(tickmode="array",tickvals=list(range(len(g))),ticktext=list(g.label),
        range=[-0.8,len(g)-0.3],showgrid=False),
    legend=dict(orientation="h",yanchor="top",y=-0.13,xanchor="center",x=0.5,font=dict(size=11)))
fig.write_image(f"{OUT}/Fig_eye_gain.png",scale=2)
fig.write_html(f"{OUT}/Fig_eye_gain.html")
print(f"\nsaved final_manuscript/figures/Fig_eye_gain.png/.html  ({len(g)} exemplar diseases)")

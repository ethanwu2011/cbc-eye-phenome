"""NHANES convergent-validation figure (2 panels):
A) Complementarity map — eye's marginal value vs CBC's marginal value per condition, quadrant-coloured.
B) Combining beats demographics — demo -> eye+CBC lollipop for the top-gain conditions."""
from config import CBCEYE_ROOT
import pandas as pd, numpy as np, plotly.graph_objects as go
from plotly.subplots import make_subplots
HERE=CBCEYE_ROOT; OUT=f"{HERE}/final_manuscript/figures"
COL={"synergy":"#2e7d52","eye-specific":"#2b6e6e","CBC-specific":"#9c5b5b","comparable":"#bdbdbd"}
r=pd.read_csv(f"{HERE}/nhanes_validation_results.csv")

# ---------- Panel A: complementarity map ----------
figA=go.Figure()
figA.add_hline(y=0.01,line=dict(color="#ddd",width=1,dash="dot")); figA.add_vline(x=0.01,line=dict(color="#ddd",width=1,dash="dot"))
for k,c in COL.items():
    s=r[r.klass==k]
    figA.add_trace(go.Scatter(x=s.cbc_adds.clip(-0.04,0.17),y=s.eye_adds,mode="markers",name=k,
        marker=dict(size=7+np.sqrt(s.n)*0.5,color=c,opacity=0.82,line=dict(width=0.5,color="#fff")),
        text=s.condition,hovertemplate="%{text}<br>CBC adds %{x:.3f} · eye adds %{y:.3f}<extra></extra>"))
LAB={"Stroke":(0,1),"Memory/confusion problem":(0,1),"Diabetes [eye=DR, expected]":(1,1),
 "Albuminuria (ACR>=30)":(1,-1),"Congestive heart failure":(1,1),"Iron deficiency (ferritin<15)":(-1,1),
 "Heart attack (MI)":(1,-1),"High CRP (inflammation)":(0,-1),"Obesity (BMI>=30)":(0,-1),"Low HDL":(0,-1)}
for _,row in r.iterrows():
    if row.condition in LAB:
        dx,dy=LAB[row.condition]; nm=row.condition.split(" [")[0].split(" (")[0]
        figA.add_annotation(x=min(row.cbc_adds,0.17),y=row.eye_adds,text=nm,showarrow=True,arrowwidth=0.6,
            arrowcolor="#999",ax=18*dx,ay=-14*dy,font=dict(size=9,color="#333"))
figA.add_annotation(x=0.13,y=0.042,text="<b>SYNERGY</b> — both add",showarrow=False,font=dict(size=11,color="#2e7d52"))
figA.add_annotation(x=-0.025,y=0.042,text="<b>eye-led</b> (neuro)",showarrow=False,font=dict(size=11,color="#2b6e6e"),xanchor="left")
figA.add_annotation(x=0.13,y=-0.05,text="<b>CBC-led</b> (inflammatory / visceral)",showarrow=False,font=dict(size=11,color="#9c5b5b"))
figA.update_layout(template="simple_white",font=dict(family="Helvetica",size=12),width=720,height=620,
    title=dict(text="<b>A   What each modality adds, by condition</b>  <sub>(NHANES, n=5,704)</sub>",x=0.02),
    xaxis=dict(title="CBC marginal value (ΔAUROC over demographics+eye)",range=[-0.045,0.175]),
    yaxis=dict(title="eye marginal value (ΔAUROC over demographics+CBC)",range=[-0.065,0.05]),
    legend=dict(orientation="h",y=-0.16,x=0.5,xanchor="center"))
figA.write_image(f"{OUT}/Fig_nhanes_map.png",scale=3)

# ---------- Panel B: combining beats demographics ----------
rb=r[(~r.condition.str.contains("circular",case=False))&(r.both<0.99)].copy()
rb["gain"]=rb.both-rb.demo; t=rb.sort_values("gain",ascending=False).head(24).iloc[::-1]
figB=go.Figure()
for _,row in t.iterrows():
    figB.add_trace(go.Scatter(x=[row.demo,row.both],y=[row.condition,row.condition],mode="lines",
        line=dict(color="#e3e3e3",width=3),showlegend=False,hoverinfo="skip"))
figB.add_trace(go.Scatter(x=t.demo,y=t.condition,mode="markers",name="demographics only",
    marker=dict(color="#bdbdbd",size=8,line=dict(width=0.5,color="#fff")),hovertemplate="demo %{x:.3f}<extra></extra>"))
figB.add_trace(go.Scatter(x=t.both,y=t.condition,mode="markers",name="eye + CBC + demographics",
    marker=dict(color=[COL[k] for k in t.klass],size=11,line=dict(width=0.5,color="#fff")),
    hovertemplate="eye+CBC %{x:.3f}<extra></extra>"))
figB.update_layout(template="simple_white",font=dict(family="Helvetica",size=11),width=720,height=620,
    title=dict(text="<b>B   Eye + CBC vs demographics alone</b>  <sub>top conditions by gain · 50/66 weighted</sub>",x=0.02),
    xaxis=dict(title="AUROC",range=[0.5,0.92]),margin=dict(l=210),
    legend=dict(orientation="h",y=-0.1,x=0.5,xanchor="center"))
figB.write_image(f"{OUT}/Fig_nhanes_combining.png",scale=3)
# compose A|B into one figure
from PIL import Image
a=Image.open(f"{OUT}/Fig_nhanes_map.png"); b=Image.open(f"{OUT}/Fig_nhanes_combining.png")
W=a.width+b.width; H=max(a.height,b.height); canvas=Image.new("RGB",(W,H),(255,255,255))
canvas.paste(a,(0,0)); canvas.paste(b,(a.width,0)); canvas.save(f"{OUT}/Fig_NHANES_validation.png")
print("saved Fig_nhanes_map.png + Fig_nhanes_combining.png + Fig_NHANES_validation.png")
print("taxonomy:",r.klass.value_counts().to_dict())

"""Fig 1B (organ-level) - readable companion to the 528-disease landscape.
Each point = one organ system at MEAN AUROC(eye) vs MEAN AUROC(CBC); bubble size = # diseases.
EQUAL square axes (true 45-deg diagonal) + a diverging gradient background: maroon above the
diagonal = CBC-led, teal below = eye-led."""
from config import CBCEYE_ROOT
import pandas as pd, numpy as np, plotly.graph_objects as go
HERE=CBCEYE_ROOT; OUT=f"{HERE}/final_manuscript/figures"
EYEC,CBCC="#2b6e6e","#9c5b5b"
L=pd.read_parquet(f"{HERE}/landscape_full.parquet")
d=L[(L.circularity_tier=="clean")&(L.klass!="no signal")].copy()
g=d.groupby("group").agg(n=("phecode","size"),eye=("auc_eye","mean"),
    cbc=("auc_cbc","mean"),both=("auc_both","mean")).reset_index()
g=g[g.n>=3].copy()
g["organ"]=[x.replace(" system","").replace(" diseases","").replace("/metabolic","/metab.") for x in g.group]

LO,HI=0.538,0.726                     # identical min/max on BOTH axes -> square, 45-deg diagonal
# diverging gradient background: z = CBC - eye (signed distance from diagonal)
gx=np.linspace(LO,HI,80); XX,YY=np.meshgrid(gx,gx); Z=YY-XX
grad=[[0.0,EYEC],[0.5,"#ffffff"],[1.0,CBCC]]

POS={"hematopoietic":"top center","neoplasms":"top center","infectious":"top right",
 "respiratory":"middle right","digestive":"middle right","endocrine/metab.":"top center",
 "dermatologic":"top center","injuries & poisonings":"bottom center","symptoms":"bottom right",
 "circulatory":"top right","genitourinary":"middle right","mental disorders":"bottom left",
 "neurological":"bottom center","musculoskeletal":"middle right"}
tpos=[POS.get(o,"top center") for o in g.organ]

fig=go.Figure()
fig.add_trace(go.Heatmap(x=gx,y=gx,z=Z,colorscale=grad,zmid=0,zmin=-(HI-LO),zmax=(HI-LO),
    opacity=0.30,showscale=False,hoverinfo="skip",zsmooth="best"))
fig.add_trace(go.Scatter(x=[LO,HI],y=[LO,HI],mode="lines",
    line=dict(color="#666",dash="dash",width=1.4),hoverinfo="skip",showlegend=False))
fig.add_trace(go.Scatter(
    x=g.eye,y=g.cbc,mode="markers+text",
    text=g.organ,textposition=tpos,textfont=dict(size=11,color="#222"),
    marker=dict(size=9+g.n*0.40,color="#33485a",opacity=0.82,line=dict(width=1,color="#fff")),
    customdata=np.stack([g.n,g.both],axis=-1),
    hovertemplate="<b>%{text}</b><br>eye %{x:.3f} · CBC %{y:.3f}<br>both %{customdata[1]:.3f} · %{customdata[0]} diseases<extra></extra>"))
fig.add_annotation(x=0.5425,y=0.7085,text="<b>CBC-led</b><br>hematologic · visceral",showarrow=False,
    font=dict(color=CBCC,size=13),align="left",xanchor="left",yanchor="top")
fig.add_annotation(x=0.712,y=0.552,text="<b>eye-led</b><br>neurological",showarrow=False,
    font=dict(color=EYEC,size=13),align="right",xanchor="right",yanchor="bottom")
fig.add_annotation(x=0.706,y=0.706,text="<i>eye = CBC</i>",showarrow=False,
    font=dict(size=10,color="#666"),textangle=-45,xanchor="right",yanchor="bottom")
fig.update_layout(template="simple_white",font=dict(family="Helvetica",size=13),
    title=dict(text="<b>The eye and the blood count split systemic disease by organ system</b>"
        "<br><sup>each point an organ system (size = # diseases); above the diagonal = CBC-led, below = eye-led</sup>",x=0.5),
    width=850,height=800,showlegend=False,
    xaxis=dict(title="mean AUROC — ophthalmic findings only",range=[LO,HI],constrain="domain"),
    yaxis=dict(title="mean AUROC — complete blood count only",range=[LO,HI],
        scaleanchor="x",scaleratio=1,constrain="domain"))
fig.write_image(f"{OUT}/Fig1B_organ_landscape.png",scale=2)
fig.write_html(f"{OUT}/Fig1B_organ_landscape.html")
print(f"saved Fig1B_organ_landscape.png/.html  ({len(g)} organs, square axes {LO}-{HI})")

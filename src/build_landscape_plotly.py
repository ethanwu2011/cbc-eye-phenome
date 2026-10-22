"""Landscape scatter (eye vs CBC AUROC, by organ system) in Plotly — hoverable.
PNG for the manuscript + interactive HTML."""
from config import CBCEYE_ROOT
import pandas as pd, plotly.express as px, plotly.graph_objects as go
HERE=CBCEYE_ROOT
L=pd.read_parquet(f"{HERE}/landscape_full.parquet")
d=L[(L.circularity_tier=="clean")&(L.klass!="no signal")].copy()
d["organ system"]=d.group; d["cases"]=d.n_cases
for c in ["auc_eye","auc_cbc","auc_both"]: d[c]=d[c].round(3)

fig=px.scatter(d,x="auc_eye",y="auc_cbc",color="organ system",size="cases",size_max=26,
    hover_name="description",
    hover_data={"klass":True,"auc_eye":True,"auc_cbc":True,"auc_both":True,"cases":True,"organ system":False},
    color_discrete_sequence=px.colors.qualitative.Safe, template="simple_white",
    labels={"auc_eye":"AUROC — eye findings only","auc_cbc":"AUROC — CBC only"})
fig.add_shape(type="line",x0=0.45,y0=0.45,x1=0.92,y1=0.92,line=dict(color="#999",dash="dash",width=1))
fig.add_annotation(x=0.50,y=0.88,text="<b>CBC-specific</b><br>haematologic / visceral",showarrow=False,font=dict(color="#9c5b5b",size=13),align="left")
fig.add_annotation(x=0.85,y=0.49,text="<b>eye-specific</b><br>neurological / microvascular",showarrow=False,font=dict(color="#2b6e6e",size=13),align="right")
fig.update_traces(marker=dict(opacity=0.78,line=dict(width=0.4,color="#444")))
fig.update_layout(title=dict(text="<b>Every condition, ranked by which modality carries it</b><br><sup>each point one of 535 systemic conditions; above the diagonal = blood-count-led, below = eye-led</sup>",x=0.5),
    width=1000,height=760,legend=dict(title="organ system",font=dict(size=10)),
    xaxis=dict(range=[0.45,0.92]),yaxis=dict(range=[0.45,0.92]))
fig.write_image(f"{HERE}/landscape_plotly.png",scale=2)
fig.write_html(f"{HERE}/landscape_plotly.html")
print(f"saved landscape_plotly.png + .html  ({len(d)} diseases)")

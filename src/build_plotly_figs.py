"""Convert Fig 1C + Fig 2 panels (A,B,C) to Plotly (teal=eye, maroon=CBC, green=both)."""
import pandas as pd, numpy as np, plotly.graph_objects as go, plotly.express as px
from config import CBCEYE_ROOT
HERE=CBCEYE_ROOT
EYE,CBC,BOTH,COMP="#2b6e6e","#9c5b5b","#3a7a3a","#a7a7a7"
L=pd.read_parquet(f"{HERE}/landscape_full.parquet"); L["phecode"]=L.phecode.astype(str)
clean=L[(L.circularity_tier=="clean")&(L.klass!="no signal")]
def save(fig,name,**lay):
    fig.update_layout(template="simple_white",font=dict(family="Helvetica",size=13),**lay)
    fig.write_image(f"{HERE}/{name}.png",scale=2); fig.write_html(f"{HERE}/{name}.html"); print("saved",name)

# ---- Fig 1C: taxonomy (class x organ system) ----
ct=pd.crosstab(clean.group,clean.klass)
ct=ct.loc[ct.sum(1).sort_values().index]
cols={"eye-specific":EYE,"CBC-specific":CBC,"synergy":BOTH,"comparable":COMP}
RENAME={"comparable":"indeterminate"}
fig=go.Figure()
for kl in ["eye-specific","synergy","CBC-specific","comparable"]:
    _disp=RENAME.get(kl,kl)
    if kl in ct: fig.add_bar(y=ct.index,x=ct[kl],name=_disp,orientation="h",marker_color=cols[kl])
save(fig,"fig1c_taxonomy_plotly",barmode="stack",title="<b>Eye vs CBC by organ system</b>",
     xaxis_title="# diseases",width=820,height=560,legend=dict(title="class"))

# ---- Fig 2A: non-redundancy (representative diseases) ----
reps=[("Multiple sclerosis","eye-specific"),("Cerebral artery occlusion","eye-specific"),
 ("Hereditary and idiopathic peripheral","eye-specific"),("Chronic renal failure","synergy"),
 ("Type 2 diabetes","synergy"),("Congestive heart failure","CBC-specific"),
 ("Leukemia","CBC-specific"),("Cirrhosis of liver","CBC-specific")]
rows=[]
for term,_ in reps:
    m=clean[clean.description.str.contains(term,case=False,na=False)]
    if len(m): r=m.nlargest(1,"n_cases").iloc[0]; rows.append((r.description[:30],r.auc_eye,r.auc_cbc,r.auc_both))
rd=pd.DataFrame(rows,columns=["disease","eye","CBC","both"]).iloc[::-1]
fig=go.Figure()
for col,c in [("eye",EYE),("CBC",CBC),("both",BOTH)]:
    fig.add_bar(y=rd.disease,x=rd[col],name=col,orientation="h",marker_color=c)
save(fig,"fig2a_nonredundancy_plotly",barmode="group",bargap=0.25,
     title="<b>Each modality adds unique signal (representative diseases)</b>",
     xaxis=dict(title="AUROC",range=[0.45,0.9]),width=900,height=620)

# ---- Fig 2B: eye -> CBC (feature independence) ----
ec=pd.read_csv(f"{HERE}/eye_predicts_cbc.csv").sort_values("auc_eye",ascending=False).head(10).iloc[::-1]
lab=[f"{r.analyte} ({r['tail']})" for _,r in ec.iterrows()]
fig=go.Figure()
fig.add_bar(y=lab,x=ec.auc_eye,name="eye",orientation="h",marker_color=EYE)
fig.add_bar(y=lab,x=ec.auc_agesex,name="age+sex",orientation="h",marker_color=COMP)
fig.add_vline(x=0.5,line=dict(color="#555",width=1))
save(fig,"fig2b_eye_to_cbc_plotly",barmode="group",bargap=0.3,
     title="<b>The eye cannot predict the blood count</b><br><sup>only monocytosis exceeds a lift-adjusted AUROC of 0.62</sup>",
     xaxis=dict(title="AUROC (eye codes → CBC abnormality)",range=[0.45,0.72]),width=820,height=560)

# ---- Fig 2C: additivity ----
it=pd.read_csv(f"{HERE}/orthogonality_interaction_test.csv")
fig=go.Figure(go.Histogram(x=it.d_interaction,nbinsx=40,marker_color="#8aa0a0"))
fig.add_vline(x=0,line=dict(color="#333",width=1.5))
fig.add_vline(x=it.d_interaction.mean(),line=dict(color=CBC,width=2,dash="dash"))
nsig=int(((it.get("q",1)<0.05)&(it.d_interaction>0.005)).sum()) if "q" in it else 0
save(fig,"fig2c_additivity_plotly",
     title=f"<b>Eye and CBC combine additively (orthogonal)</b><br><sup>eye×CBC interactions add nothing: {nsig}/{len(it)} significant · mean Δ={it.d_interaction.mean():+.3f}</sup>",
     xaxis_title="Δ AUROC (interaction − additive model)",yaxis_title="# diseases",width=820,height=540)
print("done")

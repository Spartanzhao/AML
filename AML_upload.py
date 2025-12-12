import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo
lists=[]
for i in list(adata.obs.index):
    if adata.obs['age'][i]>55:
        lists.append('old')
    else:
        lists.append('young')
adata.obs['stage']=lists



labelx='stage'
typelist=list(set(adata.obs['cell_types']))
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(set(lbs))>1:
            adatas=adatac[adatac.obs['tissue_type'].isin(['AML']), :]
            if len(adatas.obs)>300:
                tmp=adatas[adatas.obs['stage']=='old']
                exp=adatas.X
                label=list(adatas.obs['stage'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')
                cellname=cellname+'_aml_stage'
                dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                xtmp=dfs.head(20)
                xtmp=xtmp.loc[:,['Feature']]
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curve', fontsize=14)
                plt.legend(loc='lower right', fontsize=10)
                plt.grid(alpha=0.3)
                plt.savefig('AUC_{}.pdf'.format(cellname))
                dfs['cluster']=[types]*len(dfs)
                dflist.append(dfs)
            else:
                print('no3',types)
        else:
            print('no1',types)
    else:
        print('no2',types)

dfo=pd.concat(dflist)
dfo.to_csv('Shap_aml_stage_table.xls',sep='\t',index=False)
#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo
lists=[]
for i in list(adata.obs.index):
    if adata.obs['age'][i]>55:
        lists.append('old')
    else:
        lists.append('young')
adata.obs['stage']=lists

dfo=pd.read_table('Shap_aml_stage_table.xls',sep='\t')
labelx='stage'
typelist=list(set(adata.obs['cell_types']))
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            x=0
            plt.figure(figsize=(8, 6))
            adatas=adatac[adatac.obs['tissue_type'].isin(['AML']), :]
            if len(adatas.obs)>300:
                for j in toplist:
                    print(types,len(adatas.obs))
                    dfx=dfo[dfo['cluster']==types]
                    dfx=dfx.reset_index(drop=True)
                    dfx=dfx.head(j)
                    genelist=list(dfx['Feature'])
                    adatas=adatas[:, adatas.var_names.isin(genelist)]
                    exp=adatas.X
                    label=list(adatas.obs['stage'])
                    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                    cellname=types.replace(' ', '_')
                    cellname=cellname+'_aml_stage'
                    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}) top{j}')
                    x+=1
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curve', fontsize=14)
                plt.legend(loc='lower right', fontsize=10)
                plt.grid(alpha=0.3)
                plt.savefig('AUC_{}_top.pdf'.format(cellname))
        else:
            print('no1',types)
    else:
        print('no2',types)

#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo

labelx='Genotype'
typelist=list(set(adata.obs['cell_types']))
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            adatas=adatac[adatac.obs['tissue_type'].isin(['AML']), :]
            exp=adatas.X
            label=list(adatas.obs['Genotype'])
            accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
            cellname=types.replace(' ', '_')
            cellname=cellname+'_aml_genotype'
            dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
            xtmp=dfs.head(20)
            xtmp=xtmp.loc[:,['Feature']]
            xtmp.to_csv('genelist_{}.txt'.format(i),sep='\t',index=False,header=False)
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(alpha=0.3)
            plt.savefig('AUC_{}.pdf'.format(cellname))
            dfs['cluster']=[types]*len(dfs)
            dflist.append(dfs)
        else:
            print('no1',types)
    else:
        print('no2',types)

dfo=pd.concat(dflist)
dfo.to_csv('Shap_aml_genotype_table.xls',sep='\t',index=False)
#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo

dfo=pd.read_table('Shap_aml_genotype_table.xls',sep='\t')
labelx='Genotype'
typelist=list(set(adata.obs['cell_types']))
print(typelist)
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            x=0
            plt.figure(figsize=(8, 6)) 
            for j in toplist:
                adatas=adatac[adatac.obs['tissue_type'].isin(['AML']), :]
                dfx=dfo[dfo['cluster']==types]
                dfx=dfx.reset_index(drop=True)
                dfx=dfx.head(j)
                genelist=list(dfx['Feature'])
                adatas=adatas[:, adatas.var_names.isin(genelist)]
                exp=adatas.X
                label=list(adatas.obs['Genotype'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')
                cellname=cellname+'_aml_genotype'
                #dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}) top{j}')
                x+=1
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(alpha=0.3)
            plt.savefig('AUC_{}_top.pdf'.format(cellname))
        else:
            print('no1',types)
    else:
        print('no2',types)

#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo
lists=[]
for i in list(adata.obs.index):
    if adata.obs['age'][i]>55:
        lists.append('old')
    else:
        lists.append('young')
adata.obs['stage']=lists



labelx='stage'
typelist=list(set(adata.obs['cell_types']))
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(set(lbs))>1:
            adatas=adatac[adatac.obs['tissue_type'].isin(['AML','healthy']), :]
            adatas=adatas[adatas.obs['stage']=='old',:]
            if len(adatas.obs)>300:
                tmp=adatas[adatas.obs['stage']=='old']
                exp=adatas.X
                label=list(adatas.obs['tissue_type'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')
                cellname=cellname+'_aml_old'
                dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                xtmp=dfs.head(20)
                xtmp=xtmp.loc[:,['Feature']]
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curve', fontsize=14)
                plt.legend(loc='lower right', fontsize=10)
                plt.grid(alpha=0.3)
                plt.savefig('AUC_{}.pdf'.format(cellname))
                dfs['cluster']=[types]*len(dfs)
                dflist.append(dfs)
            else:
                print('no3',types)
        else:
            print('no1',types)
    else:
        print('no2',types)

dfo=pd.concat(dflist)
dfo.to_csv('Shap_aml_old_table.xls',sep='\t',index=False)
#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo
lists=[]
for i in list(adata.obs.index):
    if adata.obs['age'][i]>55:
        lists.append('old')
    else:
        lists.append('young')
adata.obs['stage']=lists

dfo=pd.read_table('Shap_aml_old_table.xls',sep='\t')
labelx='stage'
typelist=list(set(adata.obs['cell_types']))
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            x=0
            plt.figure(figsize=(8, 6))
            adatas=adatac[adatac.obs['tissue_type'].isin(['AML','healthy']), :]
            adatas=adatas[adatas.obs['stage']=='old',:]
            if len(adatas.obs)>300:
                for j in toplist:
                    print(types,len(adatas.obs))
                    dfx=dfo[dfo['cluster']==types]
                    dfx=dfx.reset_index(drop=True)
                    dfx=dfx.head(j)
                    genelist=list(dfx['Feature'])
                    adatas=adatas[:, adatas.var_names.isin(genelist)]
                    exp=adatas.X
                    label=list(adatas.obs['tissue_type'])
                    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                    cellname=types.replace(' ', '_')
                    cellname=cellname+'_aml_old'
                    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}) top{j}')
                    x+=1
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curve', fontsize=14)
                plt.legend(loc='lower right', fontsize=10)
                plt.grid(alpha=0.3)
                plt.savefig('AUC_{}_top.pdf'.format(cellname))
        else:
            print('no1',types)
    else:
        print('no2',types)

#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo

labelx='tissue_type'
typelist=list(set(adata.obs['cell_types']))
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            print('yes',types)
            adatas=adatac[adatac.obs['tissue_type'].isin(['AML', 'healthy']), :]
            tmp=adatas[adatas.obs['tissue_type']=='AML']
            print(len(tmp)/len(adatas.obs),len(adatas.obs))
            exp=adatas.X
            label=list(adatas.obs['tissue_type'])
            accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
            cellname=types.replace(' ', '_')
            dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
            xtmp=dfs.head(20)
            xtmp=xtmp.loc[:,['Feature']]
            xtmp.to_csv('genelist_{}.txt'.format(i),sep='\t',index=False,header=False)
            print('y_test,y_prob',y_test,y_prob)
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(alpha=0.3)
            plt.savefig('AUC_{}.pdf'.format(cellname))
            dfs['cluster']=[types]*len(dfs)
            dflist.append(dfs)
        else:
            print('no1',types)
    else:
        print('no2',types)

dfo=pd.concat(dflist)
dfo.to_csv('Shap_tumor_normal_table.xls',sep='\t',index=False)
#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo

dfo=pd.read_table('Shap_tumor_normal_table.xls',sep='\t')
labelx='tissue_type'
typelist=list(set(adata.obs['cell_types']))
print(typelist)
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            x=0
            plt.figure(figsize=(8, 6)) 
            for j in toplist:
                adatas=adatac[adatac.obs['tissue_type'].isin(['AML', 'healthy']), :]
                dfx=dfo[dfo['cluster']==types]
                dfx=dfx.reset_index(drop=True)
                dfx=dfx.head(j)
                genelist=list(dfx['Feature'])
                adatas=adatas[:, adatas.var_names.isin(genelist)]
                exp=adatas.X
                label=list(adatas.obs['tissue_type'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')
                #dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}) top{j}')
                x+=1
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(alpha=0.3)
            plt.savefig('AUC_{}_top.pdf'.format(cellname))
        else:
            print('no1',types)
    else:
        print('no2',types)

#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo
lists=[]
for i in list(adata.obs.index):
    if adata.obs['age'][i]>55:
        lists.append('old')
    else:
        lists.append('young')
adata.obs['stage']=lists



labelx='stage'
typelist=list(set(adata.obs['cell_types']))
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(set(lbs))>1:
            adatas=adatac[adatac.obs['tissue_type'].isin(['AML','healthy']), :]
            adatas=adatas[adatas.obs['stage']=='young',:]
            if len(adatas.obs)>300:
                tmp=adatas[adatas.obs['stage']=='young']
                exp=adatas.X
                label=list(adatas.obs['tissue_type'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')
                cellname=cellname+'_aml_young'
                dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                xtmp=dfs.head(20)
                xtmp=xtmp.loc[:,['Feature']]
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curve', fontsize=14)
                plt.legend(loc='lower right', fontsize=10)
                plt.grid(alpha=0.3)
                plt.savefig('AUC_{}.pdf'.format(cellname))
                dfs['cluster']=[types]*len(dfs)
                dflist.append(dfs)
            else:
                print('no3',types)
        else:
            print('no1',types)
    else:
        print('no2',types)

dfo=pd.concat(dflist)
dfo.to_csv('Shap_aml_young_table.xls',sep='\t',index=False)
#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo
lists=[]
for i in list(adata.obs.index):
    if adata.obs['age'][i]>55:
        lists.append('old')
    else:
        lists.append('young')
adata.obs['stage']=lists

dfo=pd.read_table('Shap_aml_young_table.xls',sep='\t')
labelx='stage'
typelist=list(set(adata.obs['cell_types']))
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            x=0
            plt.figure(figsize=(8, 6))
            adatas=adatac[adatac.obs['tissue_type'].isin(['AML','healthy']), :]
            adatas=adatas[adatas.obs['stage']=='young',:]
            if len(adatas.obs)>300:
                for j in toplist:
                    print(types,len(adatas.obs))
                    dfx=dfo[dfo['cluster']==types]
                    dfx=dfx.reset_index(drop=True)
                    dfx=dfx.head(j)
                    genelist=list(dfx['Feature'])
                    adatas=adatas[:, adatas.var_names.isin(genelist)]
                    exp=adatas.X
                    label=list(adatas.obs['tissue_type'])
                    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                    cellname=types.replace(' ', '_')
                    cellname=cellname+'_aml_young'
                    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}) top{j}')
                    x+=1
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curve', fontsize=14)
                plt.legend(loc='lower right', fontsize=10)
                plt.grid(alpha=0.3)
                plt.savefig('AUC_{}_top.pdf'.format(cellname))
        else:
            print('no1',types)
    else:
        print('no2',types)

#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo

dfo=pd.read_table('Shap_sex_table.xls',sep='\t')
labelx='tissue_type'
typelist=['Classical Monocytes','Dendritic Cells','Hematopoietic Stem and Progenitor Cells','Myeloid Progenitors']
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    print(len(adatac.obs))
    if lbs!=None:
        if len(lbs)>1:
            x=0
            plt.figure(figsize=(8, 6)) 
            for j in toplist:
                print('yes')
                adatas=adatac[adatac.obs['tissue_type'].isin(['AML']), :]
                dfx=dfo[dfo['cluster']==types]
                dfx=dfx.reset_index(drop=True)
                dfx=dfx.head(j)
                genelist=list(dfx['Feature'])
                adatas=adatas[:, adatas.var_names.isin(genelist)]
                exp=adatas.X
                label=list(adatas.obs['gender'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')
                #dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}) top{j}')
                x+=1
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(alpha=0.3)
            plt.savefig('AUC_{}_top.pdf'.format(cellname+'_gender'))
        else:
            print('no1',types)
    else:
        print('no2',types)

#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo
lists=[]
for i in list(adata.obs.index):
    if adata.obs['age'][i]>55:
        lists.append('old')
    else:
        lists.append('young')
adata.obs['stage']=lists



labelx='stage'
typelist=list(set(adata.obs['cell_types']))
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(set(lbs))>1:
            adatas=adatac[adatac.obs['tissue_type'].isin(['APL']), :]
            if len(adatas.obs)>300:
                tmp=adatas[adatas.obs['stage']=='old']
                exp=adatas.X
                label=list(adatas.obs['stage'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')
                cellname=cellname+'_apl_stage'
                dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                xtmp=dfs.head(20)
                xtmp=xtmp.loc[:,['Feature']]
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curve', fontsize=14)
                plt.legend(loc='lower right', fontsize=10)
                plt.grid(alpha=0.3)
                plt.savefig('AUC_{}.pdf'.format(cellname))
                dfs['cluster']=[types]*len(dfs)
                dflist.append(dfs)
            else:
                print('no3',types)
        else:
            print('no1',types)
    else:
        print('no2',types)

dfo=pd.concat(dflist)
dfo.to_csv('Shap_apl_stage_table.xls',sep='\t',index=False)
#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo
lists=[]
for i in list(adata.obs.index):
    if adata.obs['age'][i]>55:
        lists.append('old')
    else:
        lists.append('young')
adata.obs['stage']=lists

dfo=pd.read_table('Shap_apl_stage_table.xls',sep='\t')
labelx='stage'
typelist=list(set(adata.obs['cell_types']))
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            x=0
            plt.figure(figsize=(8, 6))
            adatas=adatac[adatac.obs['tissue_type'].isin(['APL']), :]
            if len(adatas.obs)>300:
                for j in toplist:
                    print(types,len(adatas.obs))
                    dfx=dfo[dfo['cluster']==types]
                    dfx=dfx.reset_index(drop=True)
                    dfx=dfx.head(j)
                    genelist=list(dfx['Feature'])
                    adatas=adatas[:, adatas.var_names.isin(genelist)]
                    exp=adatas.X
                    label=list(adatas.obs['stage'])
                    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                    cellname=types.replace(' ', '_')
                    cellname=cellname+'_apl_stage'
                    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}) top{j}')
                    x+=1
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curve', fontsize=14)
                plt.legend(loc='lower right', fontsize=10)
                plt.grid(alpha=0.3)
                plt.savefig('AUC_{}_top.pdf'.format(cellname))
        else:
            print('no1',types)
    else:
        print('no2',types)

#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo

labelx='tissue_type'
typelist=list(set(adata.obs['cell_types']))
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            adatas=adatac[adatac.obs['tissue_type'].isin(['APL', 'AML']), :]
            tmp=adatas[adatas.obs['tissue_type']=='AML']
            exp=adatas.X
            label=list(adatas.obs['tissue_type'])
            accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
            cellname=types.replace(' ', '_')
            cellname=cellname+'_apl_aml'
            dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
            xtmp=dfs.head(20)
            xtmp=xtmp.loc[:,['Feature']]
            xtmp.to_csv('genelist_{}.txt'.format(i),sep='\t',index=False,header=False)
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(alpha=0.3)
            plt.savefig('AUC_{}.pdf'.format(cellname))
            dfs['cluster']=[types]*len(dfs)
            dflist.append(dfs)
        else:
            print('no1',types)
    else:
        print('no2',types)

dfo=pd.concat(dflist)
dfo.to_csv('Shap_apl_aml_table.xls',sep='\t',index=False)
#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo

dfo=pd.read_table('Shap_apl_aml_table.xls',sep='\t')
labelx='tissue_type'
typelist=list(set(adata.obs['cell_types']))
print(typelist)
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            x=0
            plt.figure(figsize=(8, 6)) 
            for j in toplist:
                adatas=adatac[adatac.obs['tissue_type'].isin(['APL', 'AML']), :]
                dfx=dfo[dfo['cluster']==types]
                dfx=dfx.reset_index(drop=True)
                dfx=dfx.head(j)
                genelist=list(dfx['Feature'])
                adatas=adatas[:, adatas.var_names.isin(genelist)]
                exp=adatas.X
                label=list(adatas.obs['tissue_type'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')
                cellname=cellname+'_apl_aml'
                #dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}) top{j}')
                x+=1
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(alpha=0.3)
            plt.savefig('AUC_{}_top.pdf'.format(cellname))
        else:
            print('no1',types)
    else:
        print('no2',types)

#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def SamplePick2(adata,types):
    tissue_counts = adata.obs['tissue_type'].value_counts()
    type1, type2 = tissue_counts.index
    count1, count2 = tissue_counts.values
    if count1 > count2 * 3:
        max_type1 = count2 * 3
        max_type2 = count2
    elif count2 > count1 * 3:
        max_type1 = count1
        max_type2 = count1 * 3
    else:
        max_type1, max_type2 = count1, count2
    type1_indices = adata.obs[adata.obs['tissue_type'] == type1].index
    type2_indices = adata.obs[adata.obs['tissue_type'] == type2].index
    selected_type1_indices = np.random.choice(type1_indices, size=min(max_type1, len(type1_indices)), replace=False)
    selected_type2_indices = np.random.choice(type2_indices, size=min(max_type2, len(type2_indices)), replace=False)
    selected_indices = np.concatenate([selected_type1_indices, selected_type2_indices])
    balanced_adata = adata[selected_indices]
    balanced_counts = balanced_adata.obs['tissue_type'].value_counts()



def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    print(set(label_encoded))
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    print(set(y_train),set(y_test))
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo
lists=[]
for i in list(adata.obs.index):
    if adata.obs['age'][i]>55:
        lists.append('old')
    else:
        lists.append('young')
adata.obs['stage']=lists



labelx='tissue_type'
#typelist=list(set(adata.obs['cell_types']))
typelist=['Myeloid Progenitors','Myeloid cells']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(set(lbs))>1:
            adatas=adatac[adatac.obs['tissue_type'].isin(['APL','healthy']), :]
            adatas=adatas[adatas.obs['stage']=='old',:]
            if len(adatas.obs)>300 and len(set(adata.obs['tissue_type']))>1:
                tmp=adatas[adatas.obs['stage']=='old']
                exp=adatas.X
                label=list(adatas.obs['tissue_type'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')
                cellname=cellname+'_apl_normal_old'
                dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                xtmp=dfs.head(20)
                xtmp=xtmp.loc[:,['Feature']]
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curve', fontsize=14)
                plt.legend(loc='lower right', fontsize=10)
                plt.grid(alpha=0.3)
                plt.savefig('AUC_{}.pdf'.format(cellname))
                dfs['cluster']=[types]*len(dfs)
                dflist.append(dfs)
            else:
                print('no3',types)
        else:
            print('no1',types)
    else:
        print('no2',types)

dfo=pd.concat(dflist)
dfo.to_csv('Shap_apl_normal_old_table.xls',sep='\t',index=False)
#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo
lists=[]
for i in list(adata.obs.index):
    if adata.obs['age'][i]>55:
        lists.append('old')
    else:
        lists.append('young')
adata.obs['stage']=lists

dfo=pd.read_table('Shap_apl_normal_old_table.xls',sep='\t')
labelx='stage'
#typelist=list(set(adata.obs['cell_types']))
typelist=['Myeloid Progenitors','Myeloid cells']
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            x=0
            plt.figure(figsize=(8, 6))
            adatas=adatac[adatac.obs['tissue_type'].isin(['APL','healthy']), :]
            adatas=adatas[adatas.obs['stage']=='old',:]
            if len(adatas.obs)>300:
                for j in toplist:
                    print(types,len(adatas.obs))
                    dfx=dfo[dfo['cluster']==types]
                    dfx=dfx.reset_index(drop=True)
                    dfx=dfx.head(j)
                    genelist=list(dfx['Feature'])
                    adatas=adatas[:, adatas.var_names.isin(genelist)]
                    exp=adatas.X
                    label=list(adatas.obs['tissue_type'])
                    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                    cellname=types.replace(' ', '_')
                    cellname=cellname+'_apl_normal_old'
                    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}) top{j}')
                    x+=1
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curve', fontsize=14)
                plt.legend(loc='lower right', fontsize=10)
                plt.grid(alpha=0.3)
                plt.savefig('AUC_{}_top.pdf'.format(cellname))
        else:
            print('no1',types)
    else:
        print('no2',types)

#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo

labelx='tissue_type'
typelist=list(set(adata.obs['cell_types']))
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            print('yes',types)
            adatas=adatac[adatac.obs['tissue_type'].isin(['APL', 'healthy']), :]
            tmp=adatas[adatas.obs['tissue_type']=='AML']
            print(len(tmp)/len(adatas.obs),len(adatas.obs))
            exp=adatas.X
            label=list(adatas.obs['tissue_type'])
            accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
            cellname=types.replace(' ', '_')
            cellname=cellname+'_apl_normal'
            dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
            xtmp=dfs.head(20)
            xtmp=xtmp.loc[:,['Feature']]
            xtmp.to_csv('genelist_{}.txt'.format(i),sep='\t',index=False,header=False)
            print('y_test,y_prob',y_test,y_prob)
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(alpha=0.3)
            plt.savefig('AUC_{}.pdf'.format(cellname))
            dfs['cluster']=[types]*len(dfs)
            dflist.append(dfs)
        else:
            print('no1',types)
    else:
        print('no2',types)

dfo=pd.concat(dflist)
dfo.to_csv('Shap_apl_normal_table.xls',sep='\t',index=False)
#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo

dfo=pd.read_table('Shap_apl_normal_table.xls',sep='\t')
labelx='tissue_type'
typelist=list(set(adata.obs['cell_types']))
print(typelist)
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            x=0
            plt.figure(figsize=(8, 6)) 
            for j in toplist:
                adatas=adatac[adatac.obs['tissue_type'].isin(['APL', 'healthy']), :]
                dfx=dfo[dfo['cluster']==types]
                dfx=dfx.reset_index(drop=True)
                dfx=dfx.head(j)
                genelist=list(dfx['Feature'])
                adatas=adatas[:, adatas.var_names.isin(genelist)]
                exp=adatas.X
                label=list(adatas.obs['tissue_type'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')
                cellname=cellname+'_apl_normal'
                #dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}) top{j}')
                x+=1
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(alpha=0.3)
            plt.savefig('AUC_{}_top.pdf'.format(cellname))
        else:
            print('no1',types)
    else:
        print('no2',types)

#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def SamplePick2(adata,types):
    tissue_counts = adata.obs['tissue_type'].value_counts()
    type1, type2 = tissue_counts.index
    count1, count2 = tissue_counts.values
    if count1 > count2 * 3:
        max_type1 = count2 * 3
        max_type2 = count2
    elif count2 > count1 * 3:
        max_type1 = count1
        max_type2 = count1 * 3
    else:
        max_type1, max_type2 = count1, count2
    type1_indices = adata.obs[adata.obs['tissue_type'] == type1].index
    type2_indices = adata.obs[adata.obs['tissue_type'] == type2].index
    selected_type1_indices = np.random.choice(type1_indices, size=min(max_type1, len(type1_indices)), replace=False)
    selected_type2_indices = np.random.choice(type2_indices, size=min(max_type2, len(type2_indices)), replace=False)
    selected_indices = np.concatenate([selected_type1_indices, selected_type2_indices])
    balanced_adata = adata[selected_indices]
    balanced_counts = balanced_adata.obs['tissue_type'].value_counts()



def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    print(set(label_encoded))
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    print(set(y_train),set(y_test))
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo
lists=[]
for i in list(adata.obs.index):
    if adata.obs['age'][i]>55:
        lists.append('old')
    else:
        lists.append('young')
adata.obs['stage']=lists



labelx='tissue_type'
#typelist=list(set(adata.obs['cell_types']))
typelist=['Myeloid Progenitors','Myeloid cells']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(set(lbs))>1:
            adatas=adatac[adatac.obs['tissue_type'].isin(['APL','healthy']), :]
            adatas=adatas[adatas.obs['stage']=='young',:]
            if len(adatas.obs)>300 and len(set(adata.obs['tissue_type']))>1:
                tmp=adatas[adatas.obs['stage']=='young']
                exp=adatas.X
                label=list(adatas.obs['tissue_type'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')
                cellname=cellname+'_apl_normal_young'
                dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                xtmp=dfs.head(20)
                xtmp=xtmp.loc[:,['Feature']]
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curve', fontsize=14)
                plt.legend(loc='lower right', fontsize=10)
                plt.grid(alpha=0.3)
                plt.savefig('AUC_{}.pdf'.format(cellname))
                dfs['cluster']=[types]*len(dfs)
                dflist.append(dfs)
            else:
                print('no3',types)
        else:
            print('no1',types)
    else:
        print('no2',types)

dfo=pd.concat(dflist)
dfo.to_csv('Shap_apl_normal_young_table.xls',sep='\t',index=False)
#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo
lists=[]
for i in list(adata.obs.index):
    if adata.obs['age'][i]>55:
        lists.append('old')
    else:
        lists.append('young')
adata.obs['stage']=lists

dfo=pd.read_table('Shap_apl_normal_young_table.xls',sep='\t')
labelx='stage'
#typelist=list(set(adata.obs['cell_types']))
typelist=['Myeloid Progenitors','Myeloid cells']
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            x=0
            plt.figure(figsize=(8, 6))
            adatas=adatac[adatac.obs['tissue_type'].isin(['APL','healthy']), :]
            adatas=adatas[adatas.obs['stage']=='young',:]
            if len(adatas.obs)>300:
                for j in toplist:
                    print(types,len(adatas.obs))
                    dfx=dfo[dfo['cluster']==types]
                    dfx=dfx.reset_index(drop=True)
                    dfx=dfx.head(j)
                    genelist=list(dfx['Feature'])
                    adatas=adatas[:, adatas.var_names.isin(genelist)]
                    exp=adatas.X
                    label=list(adatas.obs['tissue_type'])
                    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                    cellname=types.replace(' ', '_')
                    cellname=cellname+'_apl_normal_young'
                    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}) top{j}')
                    x+=1
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curve', fontsize=14)
                plt.legend(loc='lower right', fontsize=10)
                plt.grid(alpha=0.3)
                plt.savefig('AUC_{}_top.pdf'.format(cellname))
        else:
            print('no1',types)
    else:
        print('no2',types)

#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo

labelx='gender'
typelist=list(set(adata.obs['cell_types']))
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(set(lbs))>1:
            adatas=adatac[adatac.obs['tissue_type'].isin(['APL']), :]
            if len(adatas.obs)>300:
                tmp=adatas[adatas.obs['gender']=='f']
                exp=adatas.X
                label=list(adatas.obs['gender'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')
                cellname=cellname+'gender_apl'
                dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                xtmp=dfs.head(20)
                xtmp=xtmp.loc[:,['Feature']]
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curve', fontsize=14)
                plt.legend(loc='lower right', fontsize=10)
                plt.grid(alpha=0.3)
                plt.savefig('AUC_{}.pdf'.format(cellname))
                dfs['cluster']=[types]*len(dfs)
                dflist.append(dfs)
            else:
                print('no3',types)
        else:
            print('no1',types)
    else:
        print('no2',types)

dfo=pd.concat(dflist)
dfo.to_csv('Shap_sex_apl_table.xls',sep='\t',index=False)
#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo

dfo=pd.read_table('Shap_sex_apl_table.xls',sep='\t')
labelx='tissue_type'
typelist=['Classical Monocytes','Dendritic Cells','Hematopoietic Stem and Progenitor Cells','Myeloid Progenitors']
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    print(len(adatac.obs))
    if lbs!=None:
        if len(lbs)>1:
            x=0
            plt.figure(figsize=(8, 6)) 
            for j in toplist:
                print('yes')
                adatas=adatac[adatac.obs['tissue_type'].isin(['APL']), :]
                dfx=dfo[dfo['cluster']==types]
                dfx=dfx.reset_index(drop=True)
                dfx=dfx.head(j)
                genelist=list(dfx['Feature'])
                adatas=adatas[:, adatas.var_names.isin(genelist)]
                exp=adatas.X
                label=list(adatas.obs['gender'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')+'gender_apl'
                #dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}) top{j}')
                x+=1
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(alpha=0.3)
            plt.savefig('AUC_{}_top.pdf'.format(cellname))
        else:
            print('no1',types)
    else:
        print('no2',types)

#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
import anndata

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


adatah=sc.read_loom('/home/dj/ZhaoYue_Home/project/AML/Data/Healthy.loom')
adatac=sc.read_loom('/home/dj/ZhaoYue_Home/project/AML/Data/AML.loom')
collist=[]
for i in list(adatah.obs.columns):
    if i in list(adatac.obs.columns):
        collist.append(i)
adatah.obs=adatah.obs.loc[:,collist]
adatac.obs=adatac.obs.loc[:,collist]
adatah.X=adatah.layers['counts']
adatac.X=adatac.layers['counts']
del adatac.layers['counts']
del adatah.layers['counts']
del adatac.layers['scale.data']
adatax=anndata.concat([adatac,adatah])

sc.pp.log1p(adatax)  # Log-transform the data
adatax.X.data[np.isnan(adatax.X.data)] = 0 
sc.pp.scale(adatax, max_value=10)
sc.tl.pca(adatax, svd_solver='arpack')
sc.pp.neighbors(adatax, n_neighbors=10, n_pcs=40)
sc.tl.umap(adatax)
sc.tl.leiden(adatax, resolution=0.5, key_added='leiden')
fig, axes = plt.subplots(2, 2, figsize=(22, 10))
sc.pl.umap(adatax, color='Batch', ax=axes[0][0],
               show=False, title="Before Batch Effect Removal")
sc.pl.umap(adatax, color='leiden', ax=axes[0][1],
               show=False, title="Before Batch Effect Removal")
reduced_data = adatax.obsm['X_pca']
metadata = adatax.obs[['Batch']].astype(str).copy()
harmony_out = hm.run_harmony(
        data_mat = reduced_data,
        meta_data = metadata,
        vars_use = ['Batch'],
        max_iter_harmony = 20,
        epsilon_harmony = 1e-4)
adatax.obsm['X_pca_harmony'] = harmony_out.Z_corr.T
sc.pp.neighbors(adatax, use_rep='X_pca_harmony')
sc.tl.umap(adatax)
sc.tl.leiden(adatax, resolution=0.5, key_added='leiden_har')
sc.pl.umap(adatax, color='Batch', ax=axes[1][0],
               show=False, title="After Batch Effect Removal")
sc.pl.umap(adatax, color='leiden_har', ax=axes[1][1],
               show=False, title="After Batch Effect Removal")
plt.tight_layout()
plt.savefig('{}_Batch_Umap.pdf'.format('AML'))

adataz=sc.read_h5ad('/home/dj/ZhaoYue_Home/project/AML/Data/Zuckerberg_AML.h5ad')
dics={}
for i in list(adataz.obs.index):
    dics[i]=adataz.obs['cell_type'][i]
lists=[]
for i in list(adatax.obs.index):
    if i in list(dics.keys()):
        lists.append(dics[i])
    else:
        lists.append('health')
adatax.obs['cell_type']=lists
fig, axes = plt.subplots(1, 1, figsize=(9, 9))
sc.pl.umap(adatax, color='cell_type', ax=axes,
               show=False, title="After Batch Effect Removal")
plt.tight_layout()
plt.savefig('{}_CellType.pdf'.format('AML'))
adatax.write_h5ad('AML_Combined.h5ad')









import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
mpl.rcParams['ytick.direction'] = 'out'
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc
from torch.nn.functional import sigmoid
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random
from sklearn.preprocessing import LabelEncoder
from kan import *

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42

def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo


dfo=pd.read_table('Shap_tumor_normal_table.xls',sep='\t')
labelx='tissue_type'
typelist=list(set(adata.obs['cell_types']))
toplist=[10]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            x=0 
            for j in toplist:
                cellname=types.replace(' ', '_')
                adatas=adatac[adatac.obs['tissue_type'].isin(['AML', 'healthy']), :]
                dfx=dfo[dfo['cluster']==types]
                dfx=dfx.reset_index(drop=True)
                dfx=dfx.head(j)
                genelist=list(dfx['Feature'])
                adatas=adatas[:, adatas.var_names.isin(genelist)]
                exp=adatas.X
                label=list(adatas.obs['tissue_type'])
                X_train, X_test, y_train, y_test = TumorClassify(exp,label)
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # Convert to float32
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Convert to long for class indices
                y_test_tensor = torch.tensor(y_test, dtype=torch.long)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                X_train_tensor = X_train_tensor.to(device)
                X_test_tensor = X_test_tensor.to(device)
                y_train_tensor = y_train_tensor.to(device)
                y_test_tensor = y_test_tensor.to(device)
                dataset={'train_input':X_train_tensor,'test_input':X_test_tensor,'train_label':y_train_tensor,'test_label':y_test_tensor}
                model = KAN(width=[10, 1],grid=50,k=5,seed=42,device=device )
                model.fit(dataset, opt="LBFGS", steps=33, lamb=0.01)
                model.plot()
                plt.savefig('KAN_tumor_normal_{}_top{}.pdf'.format(cellname,j))
                plt.close()
                logits = model(X_test_tensor)
                if logits.shape[1] == 1:  # Single output for binary classification
                    y_pred = torch.sigmoid(logits).detach().cpu().numpy()
                else:  # For multiclass classification, apply softmax
                    y_pred = F.softmax(logits, dim=1).detach().cpu().numpy()
                #y_pred = model.predict(X_test)  # Obtain predicted probabilities or scores
                print('yes')
                y_test = np.array(y_test)
                y_pred = sigmoid(logits).detach().cpu().numpy()
                y_pred = y_pred.flatten()
                fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)  # Replace "AML" with your positive class
                roc_auc = auc(fpr, tpr)
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'Receiver Operating Characteristic - Top {j}')
                plt.legend(loc='lower right')
                plt.savefig('KAN_AUC_tumor_normal_{}_top{}.pdf'.format(cellname,j))  # Save the ROC curve
                plt.close()






import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo

labelx='gender'
typelist=list(set(adata.obs['cell_types']))
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(set(lbs))>1:
            adatas=adatac[adatac.obs['tissue_type'].isin(['AML']), :]
            if len(adatas.obs)>300:
                tmp=adatas[adatas.obs['gender']=='f']
                exp=adatas.X
                label=list(adatas.obs['gender'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')
                cellname='gender_'+cellname
                dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                xtmp=dfs.head(20)
                xtmp=xtmp.loc[:,['Feature']]
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('ROC Curve', fontsize=14)
                plt.legend(loc='lower right', fontsize=10)
                plt.grid(alpha=0.3)
                plt.savefig('AUC_{}.pdf'.format(cellname))
                dfs['cluster']=[types]*len(dfs)
                dflist.append(dfs)
            else:
                print('no3',types)
        else:
            print('no1',types)
    else:
        print('no2',types)

dfo=pd.concat(dflist)
dfo.to_csv('Shap_sex_table.xls',sep='\t',index=False)
#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo

dfo=pd.read_table('Shap_sex_table.xls',sep='\t')
labelx='tissue_type'
typelist=['Classical Monocytes','Dendritic Cells','Hematopoietic Stem and Progenitor Cells','Myeloid Progenitors']
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    print(len(adatac.obs))
    if lbs!=None:
        if len(lbs)>1:
            x=0
            plt.figure(figsize=(8, 6)) 
            for j in toplist:
                print('yes')
                adatas=adatac[adatac.obs['tissue_type'].isin(['AML']), :]
                dfx=dfo[dfo['cluster']==types]
                dfx=dfx.reset_index(drop=True)
                dfx=dfx.head(j)
                genelist=list(dfx['Feature'])
                adatas=adatas[:, adatas.var_names.isin(genelist)]
                exp=adatas.X
                label=list(adatas.obs['gender'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')
                #dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}) top{j}')
                x+=1
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(alpha=0.3)
            plt.savefig('AUC_{}_top.pdf'.format(cellname+'_gender'))
        else:
            print('no1',types)
    else:
        print('no2',types)

#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr 
import harmonypy as hm

#可P
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42

h5ad='/home/dj/ZhaoYue_Home/project/AML/Analysis/Zuckerberg_AML.h5ad'
mt_tag='MT-'
batch_col='batch'
output='Zuckerberg_AML'

adata = sc.read_h5ad(h5ad)
lists=[]
for i in list(adata.var['feature_name']):
    i=i.split('_')
    if len(i)>1:
        i=i[0]
    else:
        i=i[0]
    lists.append(i)
adata.var['ENSG']=list(adata.var.index)
adata.var.index=lists

adata.obsm['X_umap'] = adata.obsm['X_umapni']
fig = plt.figure(figsize=(15,6))
axes = fig.add_axes([0.1,0.1,0.3,0.8])
sc.pl.umap(adata, color='cell_type', ax=axes,
               show=False, title="Cell Types")
handles, labels = axes.get_legend_handles_labels()  
axes.legend(handles,labels, loc="upper right", bbox_to_anchor=(3,1), ncol=2, fontsize=10,frameon=False)
axes.spines['right'].set_linewidth(0)
axes.spines['top'].set_linewidth(0)
plt.savefig('{}_CellType_Umap.pdf'.format(output))
plt.close()








import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata1 = adata[filtered_mask].copy()
    lbs=list(adata1.obs[label])
    lists=[]
    for lb in list(set(lbs)):
        adata2=adata1[adata1.obs[label]==lb,:]
        length=len(adata2.obs)
        ratio=length/len(adata1.obs)
        if length>80 and ratio>1/(len(set(lbs))*4):
            lists.append(adata2)
    if len(lists)>1:
        adata3=ad.concat(lists)
        lbs=list(adata3.obs[label])          
        return adata3,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred = le.inverse_transform(y_pred)        
    return accuracy,model,X_test,y_pred

def Shap(model,X_test,adata,y_pred,cell_type,label):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    x=0
    dflist=[]
    for i in list(set(label)):
        mean_abs_shap = np.abs(shap_values.values[:, :, x]).mean(axis=0)
        feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
        feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
        feature_importance_df = feature_importance_df.reset_index(drop=True)
        plt.figure()
        print(adata.var.index)
        shap.summary_plot(shap_values[:,:,x], X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
        plt.title("Summary Plot {} {}".format(cell_type,i))
        plt.savefig('shap_summary_{}_{}.pdf'.format(cell_type,i),bbox_inches="tight")  # Save to PDF
        plt.close()
        x+=1
        dflist.append(feature_importance_df)
    feature_importance_df=pd.concat(dflist)
    return feature_importance_df

def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('AML_Combined.h5ad')
adata=adata[adata.obs['tissue type']=='tumor',:]
lists=[]
for i in list(adata.var['feature_name']):
    i=i.split('_')
    if len(i)>1:
        i=i[0]
    else:
        i=i[0]
    lists.append(i)
adata.var['ENSG']=list(adata.var.index)
adata.var.index=lists
#sc.tl.rank_genes_groups(adata, groupby='cell_type', method='wilcoxon')
label='sex'
typelist=list(set(adata.obs['cell_type']))
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_type',label)
    if lbs!=None:
        if len(lbs)>1:
            exp=adatac.X
            types=types.replace(' ', '_')
            accuracy,model,X_test,y_pred=TumorClassify(exp,lbs)
            #print(accuracy,len(y_pred),set(y_pred),types)
            dfo=Shap(model,X_test,adatac,y_pred,types,lbs)
            dfo['cluster']=[types]*len(dfo)
            dfo=dfo.sort_values(by='Mean Absolute SHAP Value', ascending=False)
            dfo=dfo.head(10)
            dflist.append(dfo)
        else:
            print(types,len(set(lbs)),len(adatac.obs))
    else:
        print(types)

dfo.to_csv('Shap_Summary_table.xls',sep='\t',index=False)
#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc 
import anndata as ad
import pandas as pd
import numpy as np

import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


mt_tag='MT-'
output='TCGA_AML'
adata=sc.read_h5ad('TCGA_AML.h5ad')
df=pd.read_table('/home/dj/fanxn/datasests/gtf/gencode.v42.annotation.gtf',sep='\t',comment='#',header=None)
dics={}
for i in range(len(df)):
    tmp=df[8][i].split(';')
    x1=tmp[0].split('"')[1].split('.')[0]
    if tmp[2].split(' ')[1]=='gene_name':
        x2=tmp[2].split('"')[1]
        dics[x1]=x2

dfv=adata.var
lists=[]
for i in list(adata.var.index):
    i=i.split('.')[0]
    lists.append(i)
dfv['EnsemblGeneID']=lists

lists=[]
for i in list(dfv['EnsemblGeneID']):
    if i in list(dics.keys()):
        lists.append(dics[i])
    else:
        lists.append(i)
dfv['GeneName']=lists

lists=[]
for i in range(len(dfv)):
    if dfv['GeneName'][i] not in lists:
        lists.append(dfv['GeneName'][i])
    else:
        lists.append(dfv['GeneName'][i]+'-2')
dfv['gene_name']=lists

adata.var=dfv
adata.var=adata.var.set_index('gene_name')
adata.var_names_make_unique()

adata.var['mt'] = adata.var_names.str.startswith(mt_tag)  # For human datasets
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata = adata[adata.obs['pct_counts_mt'] < 20, :]
sc.pl.violin(adata, ['pct_counts_mt'], jitter=0.4, groupby=None)
plt.savefig('{}_mito_violin.pdf'.format(output))
plt.close()
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
plt.savefig('{}_mito_scatter.pdf'.format(output))

scrub = scr.Scrublet(adata.X)
doublet_scores, predicted_doublets = scrub.scrub_doublets()
# Add doublet prediction to the Scanpy object
adata.obs['doublet_score'] = doublet_scores
adata.obs['predicted_doublet'] = predicted_doublets
# Visualization of doublets
sc.pl.scatter(adata, x='total_counts', y='doublet_score', color='predicted_doublet', title="Doublet Detection")
plt.savefig('{}_doublet_scatter.png'.format(output))
plt.close()
# Remove predicted doublets & Total Counts
adata = adata[~adata.obs['predicted_doublet'], :]
sc.pl.violin(adata, keys=['total_counts'], jitter=0.4, groupby='sample', rotation=45)
plt.savefig('{}_counts_violin.pdf'.format(output))
plt.close()
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts',color='sample', title="Total Counts vs Number of Genes")
plt.savefig('{}_counts_scatter.pdf'.format(output))
plt.close()

adata = adata[(adata.obs['n_genes_by_counts'] > 200) & (adata.obs['n_genes_by_counts'] < 2500), :]
#Ribosome genes
adata.var['ribosomal'] = adata.var_names.str.startswith(('RPL', 'RPS'))
# Calculate percentage of ribosomal RNA
sc.pp.calculate_qc_metrics(adata, qc_vars=['ribosomal'], percent_top=None, log1p=False, inplace=True)
# Visualize ribosomal RNA content
sc.pl.violin(adata, keys=['pct_counts_ribosomal'], jitter=0.4, groupby=None, rotation=45)
plt.savefig('{}_ribosome_violin.pdf'.format(output))
plt.close()


adata.layers['norm']=adata.X
adata.X=adata.layers['counts']
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
adata.X.data[np.isnan(adata.X.data)] = 0 
sc.pp.log1p(adata)  # Log-transform the data

adata.X.data[np.isnan(adata.X.data)] = 0
import re
pattern = re.compile(r'^(MT-|RPL|RPS)')
genes_to_remove = adata.var_names[
    adata.var_names.str.match(pattern)]
adata = adata[:, ~adata.var_names.isin(genes_to_remove)]


sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, key_added='leiden')
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sc.pl.umap(adata, color='sample', ax=axes[0][0],
               show=False, title="Before Batch Effect Removal")
sc.pl.umap(adata, color='leiden', ax=axes[0][1],
               show=False, title="Before Batch Effect Removal")
reduced_data = adata.obsm['X_pca']
metadata = adata.obs[['sample']].astype(str).copy()
harmony_out = hm.run_harmony(
        data_mat = reduced_data,
        meta_data = metadata,
        vars_use = ['sample'],
        max_iter_harmony = 20, 
        epsilon_harmony = 1e-4)
adata.obsm['X_pca_harmony'] = harmony_out.Z_corr.T
sc.pp.neighbors(adata, use_rep='X_pca_harmony')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_har')
sc.pl.umap(adata, color='sample', ax=axes[1][0],
               show=False, title="After Batch Effect Removal")
sc.pl.umap(adata, color='leiden_har', ax=axes[1][1],
               show=False, title="After Batch Effect Removal")

for axe in axes:
    for ax in axe:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            ncol=1,
            loc='center left',  # Place legend to the right
            bbox_to_anchor=(1.01, 0.5),  # Anchor it to the right side of the plot
            fontsize='small',  # Adjust legend font size
            frameon=False)  # Optional: Remove legend box
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['right'].set_linewidth(0)
        ax.spines['top'].set_linewidth(0)
plt.savefig('TCGA_AML_UMAP.pdf')

sc.tl.rank_genes_groups(adata, groupby='leiden_har', method='wilcoxon')
marker_genes_dict = {}
for group in adata.uns['rank_genes_groups']['names'].dtype.names:
    marker_genes_dict[group] = adata.uns['rank_genes_groups']['names'][group][:30]

dics={'0':'Hematopoietic stem cells','1':'Monocytes/Macrophages','2':'Monocytes/Macrophages','3':'Hematopoietic stem cells','4':'T cells','5':'Neutrophils','6':'Hematopoietic stem cells','7':'Monocytes','8':'Neutrophils','9':'Naive T cells','10':'NK cells','11':'Hematopoietic stem cells','12':'Dendritic cells','13':'B cells'}
lists=[]
for i in list(adata.obs['leiden_har']):
    lists.append(dics[i])
adata.obs['cell_type']=lists

fig = plt.figure(figsize=(10,7))
axes = fig.add_axes([0.1,0.2,0.6,0.6])
sc.pl.umap(adata, color='cell_type', ax=axes,
               show=False, title="After Batch Effect Removal")
handles, labels = axes.get_legend_handles_labels()  # Get legend handles and labels
axes.legend(
    handles,
    labels,
    ncol=1,
    loc='center left',  # Place legend to the right
    bbox_to_anchor=(1.1, 0.5),  # Anchor it to the right side of the plot
    fontsize='small',  # Adjust legend font size
    frameon=False  # Optional: Remove legend box
)
axes.spines['bottom'].set_linewidth(1)
axes.spines['left'].set_linewidth(1)
axes.spines['right'].set_linewidth(0)
axes.spines['top'].set_linewidth(0)
plt.savefig('TCGA_AML_CellType.pdf')
plt.close()

from scipy.sparse import issparse
def count_cells_expressing_gene(adata, gene_name, threshold=0):
    if gene_name not in adata.var_names:
        raise ValueError(f"Gene '{gene_name}' not found in adata.var_names.")
    gene_expression = adata[:, gene_name].X
    if issparse(gene_expression):
        gene_expression = gene_expression.toarray()
    gene_expression = gene_expression.flatten()
    count = (gene_expression > threshold).sum()
    return count

adata.layers['scaled']=adata.X
adata.X=adata.layers['norm']

df=pd.read_table('/home/dj/areyi/AML/data/shap/shap_recurrent_vs_non_recurrent.xls',sep='\t')
df=df[df['Mean Absolute SHAP Value']>0.05]
df=df.reset_index(drop=True)
featurelist=list(df['Feature'])

genelist=[]
for gene in featurelist:
    if gene in adata.var_names:
        count=count_cells_expressing_gene(adata, gene, threshold=0)
        if count>100:
            print(count)
            genelist.append(gene)

n_genes = len(genelist)
n_cols = 5  # Number of columns
n_rows = (n_genes // n_cols) + (n_genes % n_cols > 0)  # Calculate rows needed
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = axes.flatten()

for x, gene in enumerate(genelist):
    sc.pl.umap(adata,color=gene,ax=axes[x],show=False,title=f"{gene}", color_map="viridis")
plt.savefig('TCGA_RecurrenceFeature.pdf')

df=pd.read_table('/home/dj/areyi/AML/data/shap/shap_tissue_tumor_vs_normal.xls',sep='\t')
df=df[df['Mean Absolute SHAP Value']>0.01]
df=df.reset_index(drop=True)
featurelist=list(df['Feature'])

genelist=[]
for gene in featurelist:
    if gene in adata.var_names:
        count=count_cells_expressing_gene(adata, gene, threshold=0)
        print(gene,count)
        if count>100:
            print(count)
            genelist.append(gene)

n_genes = len(genelist)
n_cols = 5  # Number of columns
n_rows = (n_genes // n_cols) + (n_genes % n_cols > 0)  # Calculate rows needed
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = axes.flatten()


for x, gene in enumerate(genelist):
    sc.pl.umap(adata,color=gene,ax=axes[x],show=False,title=f"{gene}", color_map="viridis")
plt.savefig('TCGA_TumorNormalFeature.pdf')

adata.write_h5ad('TCGA_AML_Analysis.h5ad')






import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo
lists=[]
for i in list(adata.obs.index):
    if adata.obs['tissue_type'][i]=='healthy':
        lists.append('healthy')
    else:
        lists.append('tumor')
adata.obs['tissue_type2']=lists

labelx='tissue_type'
typelist=list(set(adata.obs['cell_types']))
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            adatas=adatac[adatac.obs['tissue_type2'].isin(['tumor', 'healthy']), :]
            adatas=adatac
            tmp=adatas[adatas.obs['tissue_type']=='healthy']
            exp=adatas.X
            label=list(adatas.obs['tissue_type2'])
            accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
            cellname=types.replace(' ', '_')
            cellname=cellname+'_tumor_normal_'
            dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
            xtmp=dfs.head(20)
            xtmp=xtmp.loc[:,['Feature']]
            xtmp.to_csv('genelist_{}.txt'.format(i),sep='\t',index=False,header=False)
            print('y_test,y_prob',y_test,y_prob)
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(alpha=0.3)
            plt.savefig('AUC_tumor_normal_{}.pdf'.format(cellname))
            dfs['cluster']=[types]*len(dfs)
            dflist.append(dfs)
        else:
            print('no1',types)
    else:
        print('no2',types)

dfo=pd.concat(dflist)
dfo.to_csv('Shap_tumor_normal_table.xls',sep='\t',index=False)
#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

import sys
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
import anndata as ad
import random

from sklearn.preprocessing import LabelEncoder

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def SamplePick(adata,types,cluster,label):
    condition_type = adata.obs[cluster] == types
    filtered_mask = condition_type
    adata = adata[filtered_mask].copy()
    lbs=list(adata.obs[label])
    lists=list(set(lbs))
    if len(lists)>1 and len(adata.obs)>500:
        return adata,lbs
    else:
        return None,None

def TumorClassify(exp,label):
    le = LabelEncoder()
    label_encoded = le.fit_transform(label)
    X_train, X_test, y_train, y_test = train_test_split(exp, label_encoded, test_size=0.2, random_state=42)
    #model = xgb.XGBClassifier(objective='multi:softmax',num_class=len(set(label)), random_state=42)
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = le.inverse_transform(y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,label,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var.index),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var.index,plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df


def draw_violin(adata,groups,group_color,genelist,outfile):
    fig, axes = plt.subplots(len(genelist), 1, figsize= (len(set(adata.obs[groups])), 3 * len(genelist)), sharex=True)
    x=0
    for gene, ax in zip(genelist, axes):
        sampleLst,datalist,colorlist=adata_gene(adata,groups,group_color,gene)
        draw_ratio(ax,sampleLst, datalist, outfile, gene,colorlist)
        x+=1
    plt.savefig(outfile)

def adata_gene(adata,groups,group_color,gene):
    clusters = adata.obs[groups].cat.categories
    datalist = []
    sampleLst=[]
    # Iterate over each cluster and extract the gene expression data
    for cluster in clusters:
        # Subset adata for the current cluster
        cluster_data = adata[adata.obs[groups] == cluster]
        # Extract the expression values for the gene of interest
        # Use .X for the matrix and convert it to dense format if needed
        gene_expression = cluster_data[:, gene].X.toarray().flatten()
        # Append the gene expression values to the datalist
        datalist.append(gene_expression)
        sampleLst.append(cluster)
    colorlist=adata.uns[group_color]
    return sampleLst,datalist,colorlist

def draw_ratio(axes,sampleLst, datalist, outfile, ylabel,color_list):
    # Create the figure and axis
    axes.set_ylabel(ylabel)
    # Define tick positions and colors
    tick_list = []
    x = 1
    for i in sampleLst:
        tick_list.append(x)
        x += 1
    position = tick_list
    # Draw the violin plots
    for i in range(len(datalist)):
        x = position[i]
        data = datalist[i]

        # Create the violin plot
        bplot = axes.violinplot(
            data,
            positions=[x],
            showmeans=False,
            showextrema=False,
            widths=0.7)
        # Customize the violin body (patch)
        patch = bplot['bodies'][0]
        patch.set_facecolor(color_list[i])  # Use pre-defined color
        patch.set_alpha(0.3)               # Set transparency
        patch.set_edgecolor(color_list[i])  # Set edge color
        patch.set_linewidth(1)             # Set edge line width
    # Draw scatter points (overlaid on the violin plots)
    for i, data in enumerate(datalist):
        y = data
        x = np.random.normal(position[i], 0.1, len(y))  # Add jitter to x-coordinates
        axes.scatter(x, y, color=color_list[i], alpha=0.3, edgecolor="none")
    # Customize the plot
    axes.spines['bottom'].set_linewidth(1)
    axes.spines['left'].set_linewidth(1)
    axes.spines['right'].set_linewidth(0)
    axes.spines['top'].set_linewidth(0)
    axes.tick_params(top=False, right=False, width=1, colors='black', direction='out')
    axes.set_xticks(tick_list)
    axes.set_xticklabels(sampleLst, rotation=0, fontsize=8, ha="center")
    # Save the plot

#violin plot of clusters
adata=sc.read_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')
lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists

dfv=adata.obs
df=pd.read_table('Zuckerberg_Sample.xls',sep='\t')
df.columns=['Patient','sample','age','gender','Genotype','tissue_type']
dfo=pd.merge(dfv,df,how='left',on=['sample'])
adata.obs=dfo

lists=[]
for i in list(adata.obs.index):
    if adata.obs['tissue_type'][i]=='healthy':
        lists.append('healthy')
    else:
        lists.append('tumor')
adata.obs['tissue_type2']=lists

dfo=pd.read_table('Shap_tumor_normal_table.xls',sep='\t')
labelx='tissue_type'
typelist=list(set(adata.obs['cell_types']))
print(typelist)
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in typelist:
    adatac,lbs=SamplePick(adata,types,'cell_types',labelx)
    if lbs!=None:
        if len(lbs)>1:
            x=0
            plt.figure(figsize=(8, 6)) 
            for j in toplist:
                adatas=adatac[adatac.obs['tissue_type2'].isin(['tumor', 'healthy']), :]
                dfx=dfo[dfo['cluster']==types]
                dfx=dfx.reset_index(drop=True)
                dfx=dfx.head(j)
                genelist=list(dfx['Feature'])
                adatas=adatas[:, adatas.var_names.isin(genelist)]
                exp=adatas.X
                label=list(adatas.obs['tissue_type2'])
                accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
                cellname=types.replace(' ', '_')
                #dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}) top{j}')
                x+=1
            cellname=cellname+'_tumor_normal_'
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14)
            plt.legend(loc='lower right', fontsize=10)
            plt.grid(alpha=0.3)
            plt.savefig('AUC_{}_top.pdf'.format(cellname))
        else:
            print('no1',types)
    else:
        print('no2',types)

#df=pd.concat(dflist)
#df['cluster']=df['cluster'].astype(int)
#df=df.sort_values(by=['cluster'],ascending=True)
#df['cluster']=df['cluster'].astype(str)
#df=df.reset_index(drop=True)

"""
for types in typelist:
    tmp=df[df['cluster']==types]
    genes=list(tmp['Feature'])
    draw_violin(adata,'subcluster','subcluster_colors',genes,'shap_violin_{}.png'.format(types))
"""
import scanpy as sc 
import anndata as ad
import pandas as pd
import numpy as np

import argparse
import re,sys,os,math,gc
#plot module
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'figure.max_open_warning': 100})
plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
import scrublet as scr
import harmonypy as hm

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


mt_tag='MT-'
output='Zuckerberg_AML'
adatah=sc.read_loom('/home/dj/ZhaoYue_Home/project/AML/Data/Healthy.loom')
dfhm=pd.read_table('/home/dj/ZhaoYue_Home/project/AML/Data/Healthy_metadata.tsv')
adatac=sc.read_loom('/home/dj/ZhaoYue_Home/project/AML/Data/AML.loom')
dfcm=pd.read_table('/home/dj/ZhaoYue_Home/project/AML/Data/AML_metadata.tsv')
adatah.obs=dfhm
adatac.obs=dfcm
collist=[]
for i in list(adatah.obs.columns):
    if i in list(adatac.obs.columns):
        collist.append(i)
adatah.obs=adatah.obs.loc[:,collist]
adatac.obs=adatac.obs.loc[:,collist]
adatah.layers['norm']=adatah.X
adatac.layers['norm']=adatac.X
adatah.X=adatah.layers['counts']
adatac.X=adatac.layers['counts']
adata=ad.concat([adatac,adatah])
adata.obs['sample']=adata.obs['Batch']

df=pd.read_table('/home/dj/fanxn/datasests/gtf/gencode.v42.annotation.gtf',sep='\t',comment='#',header=None)
dics={}
for i in range(len(df)):
    tmp=df[8][i].split(';')
    x1=tmp[0].split('"')[1].split('.')[0]
    if tmp[2].split(' ')[1]=='gene_name':
        x2=tmp[2].split('"')[1]
        dics[x1]=x2

dfv=adata.var
lists=[]
for i in list(adata.var.index):
    i=i.split('.')[0]
    lists.append(i)
dfv['EnsemblGeneID']=lists

lists=[]
for i in list(dfv['EnsemblGeneID']):
    if i in list(dics.keys()):
        lists.append(dics[i])
    else:
        lists.append(i)
dfv['GeneName']=lists

lists=[]
for i in range(len(dfv)):
    if dfv['GeneName'][i] not in lists:
        lists.append(dfv['GeneName'][i])
    else:
        lists.append(dfv['GeneName'][i]+'-2')
dfv['gene_name']=lists

adata.var=dfv
adata.var_names_make_unique()

adata.var['mt'] = adata.var_names.str.startswith(mt_tag)  # For human datasets
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata = adata[adata.obs['pct_counts_mt'] < 20, :]
sc.pl.violin(adata, ['pct_counts_mt'], jitter=0.4, groupby=None)
plt.savefig('{}_mito_violin.pdf'.format(output))
plt.close()
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
plt.savefig('{}_mito_scatter.pdf'.format(output))

scrub = scr.Scrublet(adata.X)
doublet_scores, predicted_doublets = scrub.scrub_doublets()
# Add doublet prediction to the Scanpy object
adata.obs['doublet_score'] = doublet_scores
adata.obs['predicted_doublet'] = predicted_doublets
# Visualization of doublets
sc.pl.scatter(adata, x='total_counts', y='doublet_score', color='predicted_doublet', title="Doublet Detection")
plt.savefig('{}_doublet_scatter.png'.format(output))
plt.close()
# Remove predicted doublets & Total Counts
adata = adata[~adata.obs['predicted_doublet'], :]
sc.pl.violin(adata, keys=['total_counts'], jitter=0.4, groupby='sample', rotation=45)
plt.savefig('{}_counts_violin.pdf'.format(output))
plt.close()
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts',color='sample', title="Total Counts vs Number of Genes")
plt.savefig('{}_counts_scatter.pdf'.format(output))
plt.close()

adata = adata[(adata.obs['n_genes_by_counts'] > 200) & (adata.obs['n_genes_by_counts'] < 2500), :]
#Ribosome genes
adata.var['ribosomal'] = adata.var_names.str.startswith(('RPL', 'RPS'))
# Calculate percentage of ribosomal RNA
sc.pp.calculate_qc_metrics(adata, qc_vars=['ribosomal'], percent_top=None, log1p=False, inplace=True)
# Visualize ribosomal RNA content
sc.pl.violin(adata, keys=['pct_counts_ribosomal'], jitter=0.4, groupby=None, rotation=45)
plt.savefig('{}_ribosome_violin.pdf'.format(output))
plt.close()


sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
adata.X.data[np.isnan(adata.X.data)] = 0 
sc.pp.log1p(adata)  # Log-transform the data

adata.X.data[np.isnan(adata.X.data)] = 0
import re
pattern = re.compile(r'^(MT-|RPL|RPS)')
genes_to_remove = adata.var_names[
    adata.var_names.str.match(pattern)]
adata = adata[:, ~adata.var_names.isin(genes_to_remove)]

adata.obs['ClusterID']=adata.obs['ClusterID'].astype('category')
sc.pp.scale(adata, max_value=10)
sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=3000)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, key_added='leiden')
fig = plt.figure(figsize=(20,15))
ax1 = fig.add_axes([0.05,0.5,0.3,0.3])
ax2 = fig.add_axes([0.5,0.5,0.3,0.3])
sc.pl.umap(adata, color='sample', ax=ax1,
               show=False, title="Before Batch Effect Removal")
sc.pl.umap(adata, color='leiden', ax=ax2,
               show=False, title="Before Batch Effect Removal")
reduced_data = adata.obsm['X_pca']
metadata = adata.obs[['sample']].astype(str).copy()
harmony_out = hm.run_harmony(
        data_mat = reduced_data,
        meta_data = metadata,
        vars_use = ['sample'],
        max_iter_harmony = 20, 
        epsilon_harmony = 1e-4)
adata.obsm['X_pca_harmony'] = harmony_out.Z_corr.T
sc.pp.neighbors(adata, use_rep='X_pca_harmony')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_har')
ax3 = fig.add_axes([0.05,0.1,0.3,0.3])
ax4 = fig.add_axes([0.5,0.1,0.3,0.3])
sc.pl.umap(adata, color='sample', ax=ax3,
               show=False, title="After Batch Effect Removal")
sc.pl.umap(adata, color='leiden_har', ax=ax4,
               show=False, title="After Batch Effect Removal")
axlist=[ax1,ax2,ax3,ax4]
for ax in axlist:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,labels,ncol=2,loc='center left',bbox_to_anchor=(1.01, 0.5),fontsize='small', frameon=False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
plt.savefig('ZuckerBerg_AML_UMAP.pdf')

for col in adata.obs.select_dtypes(['category']).columns:
    adata.obs[col] = adata.obs[col].astype(str)

# Convert all categories in adata.var to strings
for col in adata.var.select_dtypes(['category']).columns:
    adata.var[col] = adata.var[col].astype(str)
adata.write_h5ad('ZuckerBerg_AML_Processed.h5ad')

fig = plt.figure(figsize=(26,7))
axes = fig.add_axes([0.1,0.2,0.25,0.65])
sc.pl.umap(adata, color='CellTypes', ax=axes,
               show=False, title="After Batch Effect Removal")
handles, labels = axes.get_legend_handles_labels()  # Get legend handles and labels
axes.legend(
    handles,
    labels,
    ncol=2,
    loc='center left',  # Place legend to the right
    bbox_to_anchor=(1.1, 0.5),  # Anchor it to the right side of the plot
    fontsize='small',  # Adjust legend font size
    frameon=False  # Optional: Remove legend box
)
axes.spines['bottom'].set_linewidth(1)
axes.spines['left'].set_linewidth(1)
axes.spines['right'].set_linewidth(0)
axes.spines['top'].set_linewidth(0)
plt.savefig('Zuckerberg_AML_CellType.pdf')
plt.close()

adata=sc.read_h5ad('ZuckerBerg_AML_Processed.h5ad')
dics={'CD8+ central memory T cells':'CD8+ T cells',
 'Plasma cells':'Mature and Memory B Cells',
 'Nonswitched memory B cells':'Mature and Memory B Cells',
 'HSCs & MPPs':'Hematopoietic Stem and Progenitor Cells',
 'Myelocytes':'Myeloid Progenitors',
 'CD8+ naive T cells':'CD8+ T cells',
 'CD69+PD-1+ memory CD4+ T cells':'CD4+ T cells',
 'Classical Monocytes':'Classical Monocytes',
 'Early erythroid progenitor':'Hematopoietic Stem and Progenitor Cells',
 'CD4+ cytotoxic T cells':'CD4+ T cells',
 'NK cell progenitors':'NK cells',
 'Lymphomyeloid prog':'Lymphomyeloid prog',
 'Conventional dendritic cell 1':'Dendritic Cells',
 'Late erythroid progenitor':'Erythroid cells',
 'Myeloblasts':'Myeloid Progenitors',
 'Pre-pro-B cells':'immature and Progenitor B Cells',
 'CD56brightCD16- NK cells':'NK cells',
 'NK T cells':'NKT cells',
 'Megakaryocyte progenitors':'Hematopoietic Stem and Progenitor Cells ',
 'Mature naive B cells':'Mature and Memory B Cells',
 'Non-classical monocytes':'Non-classical monocytes',
 'CD8+ effector memory T cells':'CD8+ T cells',
 'Mesenchymal cells_1':'Mesenchymal Cells',
 'Immature-like blasts':'immature and Progenitor B Cells',
 'Pre-B cells':'immature and Progenitor B Cells',
 'GammaDelta T cells':'GammaDelta T cells',
 'Erythro-myeloid progenitors':'Hematopoietic Stem and Progenitor Cells',
 'Plasmacytoid dendritic cells':'Dendritic Cells',
 'Aberrant erythroid':'Erythroid cells',
 'Class switched memory B cells':'Mature and Memory B Cells',
 'CD11c+ memory B cells':'Mature and Memory B Cells',
 'Early promyelocytes':'Myeloid Progenitors',
 'Lymphoid-primed multipotent progenitors':'Hematopoietic Stem and Progenitor Cells',
 'CD56dimCD16+ NK cells':'NK cells',
 'Eosinophil-basophil-mast cell progenitors':'Hematopoietic Stem and Progenitor Cells',
 'Promyelocytes':'Myeloid cells',
 'Conventional dendritic cell 2':'Dendritic Cells',
 'Late promyelocytes':'Myeloid cells',
 'CD8+CD103+ tissue resident memory T cells':'CD8+ T cells',
 'Mesenchymal cells_65_1':'Mesenchymal Cells',
 'Pro-B cells':'Mature and Memory B Cells',
 'Immature B cells':'immature and Progenitor B Cells',
 'CD4+ memory T cells':'CD4+ T cells',
 'Plasmacytoid dendritic cell progenitors':'Dendritic Cells'}
lists=[]
for i in list(adata.obs['CellTypes']):
    lists.append(dics[i])
adata.obs['cell_types']=lists
adata.obs
adata.write_h5ad('ZuckerBerg_Processed_ReAnno.h5ad')

lists=[]
for i in list(adata.obs['cell_types']):
    i=i.strip()
    lists.append(i)
adata.obs['cell_types']=lists
fig = plt.figure(figsize=(26,7))
axes = fig.add_axes([0.1,0.2,0.25,0.65])
sc.pl.umap(adata, color='cell_types', ax=axes,
               show=False, title="After Batch Effect Removal")
handles, labels = axes.get_legend_handles_labels()  # Get legend handles and labels
axes.legend(
    handles,
    labels,
    ncol=2,
    loc='center left',  # Place legend to the right
    bbox_to_anchor=(1.1, 0.5),  # Anchor it to the right side of the plot
    fontsize='small',  # Adjust legend font size
    frameon=False  # Optional: Remove legend box
)
axes.spines['bottom'].set_linewidth(1)
axes.spines['left'].set_linewidth(1)
axes.spines['right'].set_linewidth(0)
axes.spines['top'].set_linewidth(0)
plt.savefig('Zuckerberg_AML_cell_types.pdf')
plt.close()






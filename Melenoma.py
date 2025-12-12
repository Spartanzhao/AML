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
from scipy.sparse import issparse

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42



df1=pd.read_table('GSE78220_raw_counts_GRCh38.p13_NCBI.tsv',sep='\t')
df2=pd.read_table('GSE91061_BMS038109Sample.hg19KnownGene.raw.csv',sep=',',index_col=0)
df1=df1.set_index('GeneID')

adata1=ad.AnnData(X=df1.T)
adata2=ad.AnnData(X=df2.T)

adata1.var_names_make_unique()
adata2.var_names_make_unique()
adata1.obs_names_make_unique()
adata2.obs_names_make_unique()
adata=ad.concat([adata1,adata2])
gapdh_counts = adata[:, "GAPDH"].X
adata.layers['raw_counts']=adata.X
adata.X = adata.X / gapdh_counts.reshape(-1, 1)
adata.X = np.log2(adata.X + 1e-8)
df1o=pd.read_table('Melanoma-GSE78220.Response.tsv',sep='\t')
df2o=pd.read_table('Melanoma-GSE91061.Response.tsv',sep='\t')
columns=['sample_id', 'patient_name', 'dataset_id', 'dataset_group', 'Treatment',
       'response', 'response_NR', 'M Stage', 'overall survival (days)',
       'vital status', 'Total Mutation', 'Gender', 'Therapy', 'age_start',
       'tumor_type', 'seq_type', 'id']
df1s=pd.read_table('GSE78220.sample.tsv',sep='\t',header=None)
df1s.columns=['sample_id' ,'patient_name']
df1o = df1o.rename(columns={'sample_id': 'SRA_id'})
df1o=pd.merge(df1o,df1s,how='left',on=['patient_name'])
df1o=df1o.loc[:,columns]
df2o=df2o.loc[:,columns]
dfo=pd.DataFrame({'sample_id':list(adata.obs.index)})
print('dfo',len(dfo))
tmp=pd.concat([df1o,df2o])
print(len(tmp))
dfo=pd.merge(dfo,tmp,how='left',on=['sample_id'])
print(len(dfo))
adata.obs=dfo
print(set(adata.obs['Therapy']))
adata=adata[adata.obs['Therapy']=='anti-PD-1',:]
print(adata.X.shape)
adata.obs['judge']=list(adata.obs['response_NR'])
print(adata.X.shape)
adata=adata[adata.obs['judge']!='UNK',:]
print(adata.X.shape)
label=[]
for i in list(adata.obs['response_NR']):
    if i=='N':
        label.append(0)
    else:
        label.append(1)
adata.obs['label']=label
dfo=pd.read_table('genelist.txt',sep='\t')
genelist=list(dfo['Feature'])[:20]

tmp=adata[:, adata.var_names.isin(genelist)]
X = tmp.X.toarray() if issparse(tmp.X) else tmp.X
y = tmp.obs['label'].to_numpy()
np.save('X_train.npy', X)  # Save the data matrix
np.save('y_train.npy', y) 

device='cpu'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # Data matrix (float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Labels (long for classification)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)


dataset = {'train_input': X_train_tensor, 'test_input': X_test_tensor,
           'train_label': y_train_tensor, 'test_label': y_test_tensor}

model = KAN(width=[20,3,1, 1],grid=50,k=5,seed=42,device=device )
model.fit(dataset, opt="LBFGS", steps=33, lamb=0.003)
model.plot()
plt.savefig('Bulk_KAN_top20.pdf')
plt.close()
logits = model(X_test_tensor)
if logits.shape[1] == 1:  # Single output for binary classification
    y_pred = torch.sigmoid(logits).detach().cpu().numpy()
else:  # For multiclass classification, apply softmax
    y_pred = F.softmax(logits, dim=1).detach().cpu().numpy()
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
plt.title(f'Receiver Operating Characteristic - Top 20')
plt.legend(loc='lower right')
plt.savefig('KAN_AUC_bulk_top_20.pdf')  # Save the ROC curve
plt.close()






import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

X = np.load('X_train.npy')  # Load the data matrix
y = np.load('y_train.npy')  # Load the labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # Data matrix (float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Labels (long for classification)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_numpy = X_train_tensor.cpu().numpy()
selector = VarianceThreshold(threshold=1e-8)
X_train_filtered = selector.fit_transform(X_train_numpy)
X_train_tensor = torch.tensor(X_train_filtered, dtype=torch.float32)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_tensor.cpu().numpy())
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)

X_test_numpy = X_test_tensor.cpu().numpy()
selector = VarianceThreshold(threshold=1e-8)
X_test_filtered = selector.fit_transform(X_test_numpy)
X_test_tensor = torch.tensor(X_test_filtered, dtype=torch.float32)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test_tensor.cpu().numpy())
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)


X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

dataset = {'train_input': X_train_tensor, 'test_input': X_test_tensor,
           'train_label': y_train_tensor, 'test_label': y_test_tensor}

from kan import KAN  # Ensure you have the KAN package installed

model = KAN(width=[20,3,1, 1],grid=50,k=5,seed=42,device=device )
model.fit(dataset, opt="LBFGS", steps=33, lamb=0.003)
model.plot()
plt.savefig('KAN_bulk_top20.pdf')
plt.close()
logits = model(X_test_tensor)
if logits.shape[1] == 1:  # Single output for binary classification
    y_pred = torch.sigmoid(logits).detach().cpu().numpy()
else:  # For multiclass classification, apply softmax
    y_pred = F.softmax(logits, dim=1).detach().cpu().numpy()
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
plt.savefig('KAN_AUC_bulk_top_20.pdf')  # Save the ROC curve
plt.close()
















import pandas as pd
import patsy
import sys
import numpy.linalg as la
import numpy as np


def adjust_nums(numerical_covariates, drop_idxs):
    # if we dropped some values, have to adjust those with a larger index.
    if numerical_covariates is None: return drop_idxs
    return [nc - sum(nc < di for di in drop_idxs) for nc in numerical_covariates]

def design_mat(mod, numerical_covariates, batch_levels):
    # require levels to make sure they are in the same order as we use in the
    # rest of the script.
    design = patsy.dmatrix("~ 0 + C(batch, levels=%s)" % str(batch_levels),
                                                  mod, return_type="dataframe")

    mod = mod.drop(["batch"], axis=1)
    numerical_covariates = list(numerical_covariates)
    sys.stderr.write("found %i batches\n" % design.shape[1])
    other_cols = [c for i, c in enumerate(mod.columns)
                  if not i in numerical_covariates]
    factor_matrix = mod[other_cols]
    design = pd.concat((design, factor_matrix), axis=1)
    if numerical_covariates is not None:
        sys.stderr.write("found %i numerical covariates...\n"
                            % len(numerical_covariates))
        for i, nC in enumerate(numerical_covariates):
            cname = mod.columns[nC]
            sys.stderr.write("\t{0}\n".format(cname))
            design[cname] = mod[mod.columns[nC]]
    sys.stderr.write("found %i categorical variables:" % len(other_cols))
    sys.stderr.write("\t" + ", ".join(other_cols) + '\n')
    return design


def combat(data, batch, model=None, numerical_covariates=None):
    """Correct for batch effects in a dataset

    Parameters
    ----------
    data : pandas.DataFrame
        A (n_features, n_samples) dataframe of the expression or methylation
        data to batch correct
    batch : pandas.Series
        A column corresponding to the batches in the data, with index same as
        the columns that appear in ``data``
    model : patsy.design_info.DesignMatrix, optional
        A model matrix describing metadata on the samples which could be
        causing batch effects. If not provided, then will attempt to coarsely
        correct just from the information provided in ``batch``
    numerical_covariates : list-like
        List of covariates in the model which are numerical, rather than
        categorical

    Returns
    -------
    corrected : pandas.DataFrame
        A (n_features, n_samples) dataframe of the batch-corrected data
    """
    if isinstance(numerical_covariates, str):
        numerical_covariates = [numerical_covariates]
    if numerical_covariates is None:
        numerical_covariates = []

    if model is not None and isinstance(model, pd.DataFrame):
        model["batch"] = list(batch)
    else:
        model = pd.DataFrame({'batch': batch})

    batch_items = model.groupby("batch").groups.items()
    batch_levels = [k for k, v in batch_items]
    batch_info = [v for k, v in batch_items]
    n_batch = len(batch_info)
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(sum(n_batches))

    # drop intercept
    drop_cols = [cname for cname, inter in  ((model == 1).all()).iteritems() if inter == True]
    drop_idxs = [list(model.columns).index(cdrop) for cdrop in drop_cols]
    model = model[[c for c in model.columns if not c in drop_cols]]
    numerical_covariates = [list(model.columns).index(c) if isinstance(c, str) else c
            for c in numerical_covariates if not c in drop_cols]

    design = design_mat(model, numerical_covariates, batch_levels)

    sys.stderr.write("Standardizing Data across genes.\n")
    B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), data.T)
    grand_mean = np.dot((n_batches / n_array).T, B_hat[:n_batch,:])
    var_pooled = np.dot(((data - np.dot(design, B_hat).T)**2), np.ones((int(n_array), 1)) / int(n_array))

    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array))))
    tmp = np.array(design.copy())
    tmp[:,:n_batch] = 0
    stand_mean  += np.dot(tmp, B_hat).T

    s_data = ((data - stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, int(n_array)))))

    sys.stderr.write("Fitting L/S model and finding priors\n")
    batch_design = design[design.columns[:n_batch]]
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)

    delta_hat = []

    for i, batch_idxs in enumerate(batch_info):
        #batches = [list(model.columns).index(b) for b in batches]
        delta_hat.append(s_data[batch_idxs].var(axis=1))

    gamma_bar = gamma_hat.mean(axis=1) 
    t2 = gamma_hat.var(axis=1)
    a_prior = list(map(aprior, delta_hat))
    b_prior = list(map(bprior, delta_hat))

    sys.stderr.write("Finding parametric adjustments\n")
    gamma_star, delta_star = [], []
    for i, batch_idxs in enumerate(batch_info):
        #print '18 20 22 28 29 31 32 33 35 40 46'
        #print batch_info[batch_id]

        temp = it_sol(s_data[batch_idxs], gamma_hat[i],
                     delta_hat[i], gamma_bar[i], t2[i], a_prior[i], b_prior[i])

        gamma_star.append(temp[0])
        delta_star.append(temp[1])

    sys.stdout.write("Adjusting data\n")
    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)


    for j, batch_idxs in enumerate(batch_info):

        dsq = np.sqrt(delta_star[j,:])
        dsq = dsq.reshape((len(dsq), 1))
        denom =  np.dot(dsq, np.ones((1, n_batches[j])))
        numer = np.array(bayesdata[batch_idxs] - np.dot(batch_design.loc[batch_idxs], gamma_star).T)

        bayesdata[batch_idxs] = numer / denom
   
    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array)))) + stand_mean
 
    return bayesdata


def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    n = (1 - np.isnan(sdat)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change = 1
    count = 0
    while change > conv:
        #print g_hat.shape, g_bar.shape, t2.shape
        g_new = postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((sdat - np.dot(g_new.values.reshape((g_new.shape[0], 1)), np.ones((1, sdat.shape[1])))) ** 2).sum(axis=1)
        d_new = postvar(sum2, n, a, b)
       
        change = max((abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max())
        g_old = g_new #.copy()
        d_old = d_new #.copy()
        count = count + 1
    adjust = (g_new, d_new)
    return adjust 

    

def aprior(gamma_hat):
    m = gamma_hat.mean()
    s2 = gamma_hat.var()
    return (2 * s2 +m**2) / s2

def bprior(gamma_hat):
    m = gamma_hat.mean()
    s2 = gamma_hat.var()
    return (m*s2+m**3)/s2

def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

if __name__ == "__main__":
    # NOTE: run this first to get the bladder batch stuff written to files.
    """
    source("http://bioconductor.org/biocLite.R")
    biocLite("sva")

	library("sva")
	options(stringsAsFactors=FALSE)

	library(bladderbatch)
	data(bladderdata)

	pheno = pData(bladderEset)
	# add fake age variable for numeric
	pheno$age = c(1:7, rep(1:10, 5))
        write.table(data.frame(cel=rownames(pheno), pheno), row.names=F, quote=F, sep="\t", file="bladder-pheno.txt")

	edata = exprs(bladderEset)
    write.table(edata, row.names=T, quote=F, sep="\t", file="bladder-expr.txt")
	# use dataframe instead of matrix
	mod = model.matrix(~as.factor(cancer) + age, data=pheno)
    t = Sys.time()
	cdata = ComBat(dat=edata, batch=as.factor(pheno$batch), mod=mod, numCov=match("age", colnames(mod)))
    print(Sys.time() - t)
    print(cdata[1:5, 1:5])
    write.table(cdata, row.names=True, quote=F, sep="\t", file="r-batch.txt")
    """

    pheno = pd.read_table('bladder-pheno.txt', index_col=0)
    dat = pd.read_table('bladder-expr.txt', index_col=0)

    mod = patsy.dmatrix("~ age + cancer", pheno, return_type="dataframe")
    import time
    t = time.time()
    ebat = combat(dat, pheno['batch'], mod, "age")
    sys.stdout.write("%.2f seconds\n" % (time.time() - t))

    sys.stdout.write(str(ebat.iloc[:5, :5]))

    ebat.to_csv("py-batch.txt", sep="\t")

    ebat = combat(dat, pheno['batch'], None)


import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# 设置 scanpy 日志级别
sc.settings.verbosity = 3

# 1. 加载响应元数据 ====================================================
def load_response_data():
    # 加载三个数据集的响应元数据
    gse78220_response = pd.read_csv('Melanoma-GSE78220.Response.tsv', sep='\t')
    gse93157_response = pd.read_csv('Melanoma-GSE93157.Response.tsv', sep='\t')
    gse91061_response = pd.read_csv('Melanoma-GSE91061.Response.tsv', sep='\t')
    
    # 统一列名（确保所有数据集都有sample_id列）
    gse91061_response = gse91061_response.rename(columns={'Run_s':'sample_id'})
    
    # 统一筛选条件
    for df in [gse78220_response, gse93157_response, gse91061_response]:
        df['Therapy'] = df['Therapy'].str.lower()  # 统一小写处理
    
    # 筛选 anti-PD-1 治疗且响应为 R/N 的样本
    dfs = []
    for df in [gse78220_response, gse93157_response, gse91061_response]:
        filtered = df[(df['Therapy'] == 'anti-pd-1') & 
                     (df['response_NR'].isin(['R', 'N']))].copy()
        filtered['response_NR'] = filtered['response_NR'].astype('category')
        dfs.append(filtered)
    
    return dfs

# 2. 加载计数数据 =====================================================
def load_count_data():
    # 加载三个数据集的计数数据
    gse78220_counts = pd.read_csv('GSE78220_raw_counts_GRCh38.p13_NCBI.tsv', 
                                 sep='\t', index_col=0)
    gse93157_counts = pd.read_csv('GSE93157_raw_data_values.txt', 
                                 sep='\t', index_col=0)
    gse91061_counts = pd.read_csv('GSE91061_BMS038109Sample.hg19KnownGene.raw.csv',
                                 index_col=0)
    
    # 统一GSE91061的列名格式（SRR样本ID）
    gse91061_counts.columns = [col.split('.')[0] for col in gse91061_counts.columns]
    
    return gse78220_counts, gse93157_counts, gse91061_counts

# 3. DESeq2 差异表达分析函数 ===========================================
def run_deseq2(counts_df, metadata_df, group_col, group1, group2, output_count_file):
    # 确保计数数据只包含非零表达基因
    counts_df = counts_df[counts_df.sum(axis=1) > 0]
    
    # 保存计数数据到临时文件
    counts_df.to_csv('temp_counts.tsv', sep='\t')
    
    # 准备元数据
    metadata = metadata_df[[group_col]].copy()
    metadata = metadata.reset_index()
    metadata.columns = ['sample_id', group_col]
    metadata = metadata.set_index('sample_id')
    metadata[group_col] = metadata[group_col].astype(str)
    metadata.to_csv('temp_metadata.tsv', sep='\t')
    
    # 运行 DESeq2
    counts = pd.read_csv('temp_counts.tsv', sep='\t', index_col=0)
    counts = counts.T  # DESeq2 需要样本为行
    metadata = pd.read_csv('temp_metadata.tsv', sep='\t', index_col=0)
    
    dds = DeseqDataSet(counts=counts, metadata=metadata, design_factors=[group_col])
    dds.deseq2()
    
    stat_res = DeseqStats(dds, contrast=[group_col, group1, group2])
    stat_res.summary()
    res = stat_res.results_df
    
    # 获取归一化计数
    norms = stat_res.dds.layers['normed_counts']
    norms = pd.DataFrame(norms, index=counts.index, columns=counts.columns)
    norms = norms.T  # 转回基因为行
    norms['Geneid'] = norms.index
    norms = norms.reset_index(drop=True)
    
    # 保存归一化计数
    norms.to_csv(output_count_file, index=False)
    
    # 准备结果
    res['Geneid'] = res.index
    res = res.reset_index(drop=True)
    dfo = pd.merge(res, norms, on=['Geneid'], how='inner')
    
    return dfo

# 4. 火山图函数 =======================================================
def plot_volcano(df, title, output_file, top_n=20):
    df['-log10_AdjP'] = -np.log10(df['padj'].clip(lower=1e-300))
    logfc_thresh = 1.0
    pval_thresh = 0.05
    df['Significant'] = ((df['log2FoldChange'].abs() > logfc_thresh) & 
                        (df['padj'] < pval_thresh))
    
    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(
        data=df, x='log2FoldChange', y='-log10_AdjP',
        hue='Significant', palette={True: 'red', False: 'gray'},
        alpha=0.6, s=40
    )
    
    # 标记top基因
    top_genes = df[df['Significant']].nsmallest(top_n, 'padj')
    for _, row in top_genes.iterrows():
        ax.text(row['log2FoldChange'], row['-log10_AdjP'], row['Geneid'], 
                fontsize=10, ha='center', va='bottom')
    
    plt.axhline(-np.log10(pval_thresh), linestyle='--', color='black')
    plt.axvline(logfc_thresh, linestyle='--', color='black')
    plt.axvline(-logfc_thresh, linestyle='--', color='black')
    plt.title(f"{title} (n={df.shape[0]})", fontsize=14)
    plt.xlabel('Log2 Fold Change', fontsize=12)
    plt.ylabel('-log10(Adjusted P-value)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

# 5. 主分析函数 ========================================================
def main():
    # 加载数据
    response_dfs = load_response_data()
    count_dfs = load_count_data()
    
    # 获取所有数据集的名称
    dataset_names = ['GSE78220', 'GSE93157', 'GSE91061']
    
    # 打印各数据集信息
    for name, df in zip(dataset_names, response_dfs):
        print(f"\n{name} 响应元数据（筛选后）：")
        print(f"样本数量：{len(df)}")
        print("response_NR 分组统计：")
        print(df['response_NR'].value_counts())
        print("sample_id（前5个）：", df['sample_id'].head().tolist())
    
    # 获取基因交集
    common_genes = reduce(
        lambda x, y: x.intersection(y),
        [df.index for df in count_dfs]
    )
    print(f"\n共同基因数量：{len(common_genes)}")
    
    # 筛选各数据集的匹配样本
    matched_samples = []
    for df, counts in zip(response_dfs, count_dfs):
        samples = set(df['sample_id']).intersection(counts.columns)
        matched_samples.append(samples)
        print(f"\n{dataset_names[len(matched_samples)-1]} 匹配样本数：{len(samples)}")
    
    # 合并所有响应数据
    response_df = pd.concat([
        df[df['sample_id'].isin(samples)] 
        for df, samples in zip(response_dfs, matched_samples)
    ], ignore_index=True)
    
    # 合并所有计数数据（只保留共同基因和匹配样本）
    counts_list = []
    for counts, samples in zip(count_dfs, matched_samples):
        counts = counts.loc[common_genes, list(samples)]
        counts_list.append(counts)
    
    counts_df = pd.concat(counts_list, axis=1)
    
    # 检查样本顺序一致性
    print("\n=== 样本顺序验证 ===")
    print("Counts 数据样本数:", len(counts_df.columns))
    print("Response 数据样本数:", len(response_df))
    
    # 确保response_df包含所有counts_df的样本
    missing_samples = set(counts_df.columns) - set(response_df['sample_id'])
    if missing_samples:
        print(f"警告：{len(missing_samples)}个样本在计数数据中但不在响应数据中")
        counts_df = counts_df.drop(columns=list(missing_samples))
    
    # 重新对齐数据
    response_df = response_df[response_df['sample_id'].isin(counts_df.columns)]
    response_df = response_df.sort_values('sample_id')
    counts_df = counts_df[sorted(counts_df.columns)]
    
    print("\n最终样本顺序验证：")
    print("Counts 列名前5个:", counts_df.columns.tolist()[:5])
    print("Response sample_id前5个:", response_df['sample_id'].head().tolist())
    print("顺序是否一致:", (counts_df.columns == response_df['sample_id']).all())
    
    # 检查分组样本量
    print("\n最终 response_NR 分组统计：")
    print(response_df['response_NR'].value_counts())
    if response_df['response_NR'].value_counts().min() < 2:
        raise ValueError("R 或 N 组样本量不足（少于2个）！")
    
    # 运行 DESeq2 分析
    print("\n正在进行 DESeq2 差异表达分析...")
    dea_df = run_deseq2(
        counts_df=counts_df,
        metadata_df=response_df.set_index('sample_id'),
        group_col='response_NR',
        group1='R',
        group2='N',
        output_count_file='combined_normalized_counts.csv'
    )
    
    # 生成火山图
    plot_title = '(Differential gene of anti-PD-1 treatment response , DESeq2)'
    plot_volcano(dea_df, plot_title, 'combined_volcano_plot_deseq2.png')
    
    # 输出显著基因
    sig_genes = dea_df[dea_df['padj'] < 0.05].sort_values('padj')
    print("\n显著差异表达基因（FDR < 0.05, 按p值排序）：")
    print(sig_genes.head(50).to_string())
    
    # 生成表达方向表格
    print("\n生成表达方向表格...")
    direction_df = sig_genes[['Geneid', 'log2FoldChange', 'padj']].copy()
    direction_df['Expression_Direction'] = direction_df['log2FoldChange'].apply(
        lambda x: 'R_vs_N_Higher' if x > 0 else 'R_vs_N_Lower'
    )
    direction_df = direction_df.sort_values('padj')
    direction_df.to_csv('combined_expression_direction.csv', index=False)
    print("\n表达方向表格（前10行）：")
    print(direction_df.head(10).to_string())
    
    # 保存结果
    output_files = {
        '差异表达结果': 'combined_differential_expression_results_deseq2.csv',
        '归一化数据': 'combined_normalized_counts.csv',
        '预处理计数矩阵': 'combined_raw_counts.csv',
        '表达方向表格': 'combined_expression_direction.csv'
    }
    
    dea_df.to_csv(output_files['差异表达结果'], index=False)
    counts_df.to_csv(output_files['预处理计数矩阵'])
    
    print("\n分析完成！结果已保存到：")
    for desc, path in output_files.items():
        print(f"- {desc}: {path}")

if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

# 设置 scanpy 日志级别
sc.settings.verbosity = 3

# 加载响应元数据
gse78220_response = pd.read_csv('Melanoma-GSE78220.Response.tsv', sep='\t')
gse93157_response = pd.read_csv('Melanoma-GSE93157.Response.tsv', sep='\t')

# 筛选 Therapy 为 anti-PD-1 的样本
gse78220_response = gse78220_response[gse78220_response['Therapy'] == 'anti-PD-1']
gse93157_response = gse93157_response[gse93157_response['Therapy'] == 'anti-PD-1']

# 筛选 response_NR 为 R 或 N 的样本
gse78220_response = gse78220_response[gse78220_response['response_NR'].isin(['R', 'N'])]
gse93157_response = gse93157_response[gse93157_response['response_NR'].isin(['R', 'N'])]

# 将 response_NR 转换为 category 类型
gse78220_response['response_NR'] = gse78220_response['response_NR'].astype('category')
gse93157_response['response_NR'] = gse93157_response['response_NR'].astype('category')

# 调试：打印响应元数据的分组统计和 sample_id
print("\nGSE78220 响应元数据（筛选后）：")
print(f"样本数量：{len(gse78220_response)}")
print("response_NR 分组统计：")
print(gse78220_response['response_NR'].value_counts())
print("sample_id（前 5 个）：")
print(gse78220_response['sample_id'].head().to_list())

print("\nGSE93157 响应元数据（筛选后）：")
print(f"样本数量：{len(gse93157_response)}")
print("response_NR 分组统计：")
print(gse93157_response['response_NR'].value_counts())
print("sample_id（前 5 个或全部）：")
print(gse93157_response['sample_id'].head().to_list())

# 加载计数数据
gse78220_counts = pd.read_csv('GSE78220_raw_counts_GRCh38.p13_NCBI.tsv', sep='\t', index_col=0)
gse93157_counts = pd.read_csv('GSE93157_raw_data_values.txt', sep='\t', index_col=0)

# 调试：打印计数数据的列名
print("\nGSE78220 计数数据的列名（前 5 个）：")
print(gse78220_counts.columns[:5].to_list())
print("\nGSE93157 计数数据的列名（前 5 个）：")
print(gse93157_counts.columns[:5].to_list())

# 取基因交集
common_genes = gse78220_counts.index.intersection(gse93157_counts.index)
gse78220_counts = gse78220_counts.loc[common_genes]
gse93157_counts = gse93157_counts.loc[common_genes]

# 筛选匹配的样本
gse78220_samples = set(gse78220_response['sample_id']).intersection(gse78220_counts.columns)
gse93157_samples = set(gse93157_response['sample_id']).intersection(gse93157_counts.columns)

# 调试：打印匹配的样本
print("\nGSE78220 匹配的样本 ID：")
print(list(gse78220_samples))
print(f"匹配的样本数量：{len(gse78220_samples)}")
print("\nGSE93157 匹配的样本 ID：")
print(list(gse93157_samples))
print(f"匹配的样本数量：{len(gse93157_samples)}")

# 检查样本 ID 重复
overlap_samples = gse78220_samples.intersection(gse93157_samples)
print("\nGSE78220 和 GSE93157 的重复样本 ID：")
print(list(overlap_samples))
print(f"重复样本数量：{len(overlap_samples)}")

# 合并样本 ID
total_samples = gse78220_samples.union(gse93157_samples)
print("\n总匹配的样本 ID：")
print(list(total_samples))
print(f"总样本数量：{len(total_samples)}")

# 检查样本量
if len(total_samples) < 2:
    raise ValueError("总样本量不足！请检查响应元数据和计数数据的样本 ID 匹配情况。")

# 合并计数数据
gse78220_counts = gse78220_counts[list(gse78220_samples.intersection(gse78220_counts.columns))]
gse93157_counts = gse93157_counts[list(gse93157_samples.intersection(gse93157_counts.columns))]
counts_df = pd.concat([gse78220_counts, gse93157_counts], axis=1)

# 合并响应元数据
response_df = pd.concat([gse78220_response[gse78220_response['sample_id'].isin(gse78220_samples)],
                        gse93157_response[gse93157_response['sample_id'].isin(gse93157_samples)]],
                       ignore_index=True)

# 确保 response_df 只包含 counts_df 的样本
response_df = response_df[response_df['sample_id'].isin(counts_df.columns)]

# 调试：打印 counts_df 和 response_df 的样本 ID
print("\ncounts_df 的样本 ID：")
print(counts_df.columns.to_list())
print(f"counts_df 样本数量：{len(counts_df.columns)}")
print("\nresponse_df 的样本 ID：")
print(response_df['sample_id'].to_list())
print(f"response_df 样本数量：{len(response_df)}")

# 调试：打印最终的 response_NR 分组统计
print("\n最终用于分析的 response_NR 分组统计：")
print(response_df['response_NR'].value_counts())

# 检查 R 和 N 组样本量
if response_df['response_NR'].value_counts().min() < 2:
    raise ValueError("R 或 ül组样本量不足（少于 2 个）！请检查响应元数据的分组情况。")

# 创建 AnnData 对象，确保索引对齐
adata = sc.AnnData(X=counts_df.T, obs=response_df.set_index('sample_id').reindex(counts_df.columns))

# 数据预处理（归一化）
sc.pp.normalize_total(adata, target_sum=1e6)  # CPM 归一化
sc.pp.log1p(adata)  # log 转换

# 差异基因分析
sc.tl.rank_genes_groups(adata, groupby='response_NR', method='wilcoxon', groups=['R'], reference='N')

# 提取 DGE 结果
dea_results = []
for group in adata.uns['rank_genes_groups']['names'].dtype.names:
    genes = adata.uns['rank_genes_groups']['names'][group]
    logfoldchanges = adata.uns['rank_genes_groups']['logfoldchanges'][group]
    pvals_adj = adata.uns['rank_genes_groups']['pvals_adj'][group]
    for gene, lfc, pval in zip(genes, logfoldchanges, pvals_adj):
        dea_results.append({'Gene': gene, 'Log2FoldChange': lfc, 'AdjustedPValue': pval})

dea_df = pd.DataFrame(dea_results)

# 火山图函数
def plot_volcano(dea_df, title, output_file, top_n=10):
    dea_df['-log10_AdjP'] = -np.log10(dea_df['AdjustedPValue'].clip(lower=1e-300))
    logfc_threshold = 1.0
    pval_threshold = 0.05
    dea_df['Significant'] = ((dea_df['Log2FoldChange'].abs() > logfc_threshold) & 
                            (dea_df['AdjustedPValue'] < pval_threshold))
    top_genes = dea_df[dea_df['Significant']].sort_values('AdjustedPValue').head(top_n)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=dea_df, x='Log2FoldChange', y='-log10_AdjP', 
                    hue='Significant', size='Significant', sizes=(20, 100),
                    palette={False: '#d3d3d3', True: '#ff4500'}, alpha=0.7)
    for _, row in top_genes.iterrows():
        plt.text(row['Log2FoldChange'], row['-log10_AdjP'], row['Gene'], 
                 fontsize=10, ha='center', va='bottom')
    plt.axhline(y=-np.log10(pval_threshold), color='black', linestyle='--', linewidth=1)
    plt.axvline(x=logfc_threshold, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=-logfc_threshold, color='black', linestyle='--', linewidth=1)
    plt.title(title, fontsize=14, pad=10)
    plt.xlabel('Log2 Fold Change', fontsize=12)
    plt.ylabel('-log10(Adjusted P-value)', fontsize=12)
    plt.legend(title='Significant', loc='upper right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


    
# 生成火山图
plot_volcano(dea_df, 'Differential Gene Expression: Response vs Non-Response', 'volcano_plot.png')

# 输出前 50 个显著基因
significant_genes = dea_df[dea_df['AdjustedPValue'] < 0.05].sort_values('AdjustedPValue').head(50)
for _, row in significant_genes.iterrows():
    print(f"Gene: {row['Gene']}, Log2FoldChange: {row['Log2FoldChange']:.4f}, AdjustedPValue: {row['AdjustedPValue']:.4e}")

# 保存 DGE 结果到 CSV
dea_df.to_csv('dge_results.csv', index=False)
import streamlit as st
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
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
#plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
from io import BytesIO

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def deseq2(count,meta,group1,group2,factors):
    counts = pd.read_table(count,sep='\t',index_col=0)
    counts = counts[counts.sum(axis = 1) > 0]    
    counts = counts.T
    metadata = pd.read_table(meta,sep='\t')
    metadata = metadata.fillna('None')
    for i in factors:
        metadata[i]=metadata[i].astype(str)
    metadata = metadata.set_index(list(metadata.columns)[0])
    dds = DeseqDataSet(counts=counts,metadata=metadata,design_factors=factors)
    dds.deseq2()
    stat_res = DeseqStats(dds, contrast = [factors[0],group1,group2])
    stat_res.summary()
    res = stat_res.results_df
    norms=stat_res.dds.layers['normed_counts']
    norms=pd.DataFrame(norms)
    norms.columns=list(counts.columns)
    norms=norms.T
    norms.columns=list(counts.index)
    cols=metadata[(metadata[factors[0]]==group1)|(metadata[factors[0]]==group2)]
    cols=list(cols.index)
    norms=norms.loc[:,cols]
    norms['Geneid']=list(norms.index)
    norms=norms.reset_index(drop=True)
    res['Geneid']=list(res.index)
    res=res.reset_index(drop=True)
    dfo=pd.merge(res,norms,on=['Geneid'],how='inner')
    #dfo2=pd.merge(res,norms,on=['Geneid'],how='inner')
    return dfo

def vocalno(data,fold_change_threshold, pvalue_threshold,vals,xlim,ylim):
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_axes([0.13,0.1,0.5,0.7])
    fold_change_threshold = fold_change_threshold # Absolute log2 fold change > 1
    pvalue_threshold = pvalue_threshold    # p-value < 0.05
    data["Significance"] = "Not Significant"
    data.loc[data[vals]<pvalue_threshold,"Significance"]="Significant"
    data.loc[(data["log2FoldChange"] > fold_change_threshold) & (data[vals] < pvalue_threshold), "Significance"] = "Significant up"
    data.loc[(data["log2FoldChange"] < -fold_change_threshold) & (data[vals] < pvalue_threshold), "Significance"] = "Significant down"
    data["-log10(pvalue)"] = -np.log10(data[vals])
    colors = {"Significant up": "#EC0000", "Not Significant": "#DBDBDB","Significant down":"#0080F6","Significant":"#FCECCC"}
    data=data.reset_index(drop=True)
    draw_colors=[]
    for i in range(len(data)):
        draw_colors.append(colors[data['Significance'][i]])
    plt.scatter(data["log2FoldChange"], data["-log10(pvalue)"], c=draw_colors, alpha=0.5,edgecolors='none')
    
    # Add horizontal and vertical lines for thresholds
    plt.axhline(-np.log10(pvalue_threshold), color="blue", linestyle="--", linewidth=1, label="p-value threshold")
    plt.axvline(-fold_change_threshold, color="green", linestyle="--", linewidth=1, label="log2FC threshold")
    plt.axvline(fold_change_threshold, color="green", linestyle="--", linewidth=1)
    plt.xlim(-xlim,xlim)
    plt.ylim(0,ylim)
    for label, color in colors.items():
        plt.scatter([], [], color=color, label=label)
    # Titles and labels
    plt.title("Volcano Plot")
    plt.xlabel("log2(Fold Change)")
    plt.ylabel("-log10(p-value)")
    plt.legend(frameon=False, loc="upper right",bbox_to_anchor=(1.7, 1))
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    plt.grid(alpha=0)
    plt.tight_layout()
    return plt


count_file = 'Melenoma_CountPre.txt'
meta_file = 'Melenoma_ObsPre.txt'
group1 = 'R'
group2 = 'N'
vals = "pvalue"
fold_change_threshold = 1
pvalue_threshold = 0.05
xlim=3.5
ylim=12
factors=['response_NR','dataset_id']

dfo = deseq2(count_file, meta_file, group1, group2,factors)
dfo.to_csv('Volcano_table_Pre.xls',sep='\t',index=False)
plt = vocalno(dfo, fold_change_threshold, pvalue_threshold,vals,xlim,ylim)
plt.savefig('TestVolcano_Pre.pdf')





import streamlit as st
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from adjustText import adjust_text
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
#plt.style.use('seaborn-colorblind')
mpl.rcParams['ytick.direction'] = 'out'
from io import BytesIO

#可P
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42

def deseq2(count,meta,group1,group2,factors):
    counts = pd.read_table(count,sep='\t',index_col=0)
    counts = counts[counts.sum(axis = 1) > 0]
    counts = counts.T
    metadata = pd.read_table(meta,sep='\t')
    metadata = metadata.fillna('None')
    for i in factors:
        metadata[i]=metadata[i].astype(str)
    metadata = metadata.set_index(list(metadata.columns)[0])
    dds = DeseqDataSet(counts=counts,metadata=metadata,design_factors=factors)
    dds.deseq2()
    stat_res = DeseqStats(dds, contrast = [factors[0],group1,group2])
    stat_res.summary()
    res = stat_res.results_df
    norms=stat_res.dds.layers['normed_counts']
    norms=pd.DataFrame(norms)
    norms.columns=list(counts.columns)
    norms=norms.T
    norms.columns=list(counts.index)
    cols=metadata[(metadata[factors[0]]==group1)|(metadata[factors[0]]==group2)]
    cols=list(cols.index)
    norms=norms.loc[:,cols]
    norms['Geneid']=list(norms.index)
    norms=norms.reset_index(drop=True)
    res['Geneid']=list(res.index)
    res=res.reset_index(drop=True)
    dfo=pd.merge(res,norms,on=['Geneid'],how='inner')
    #dfo2=pd.merge(res,norms,on=['Geneid'],how='inner')
    return dfo


def vocalno(data,fold_change_threshold, pvalue_threshold,vals,text,gene_col,xlim,ylim):
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_axes([0.13,0.1,0.5,0.7])
    fold_change_threshold = fold_change_threshold  # Absolute log2 fold change > 1
    pvalue_threshold = pvalue_threshold    # p-value < 0.05
    data["Significance"] = "Not Significant"
    data.loc[data[vals]<pvalue_threshold,"Significance"]="Significant"
    data.loc[(data["log2FoldChange"] > fold_change_threshold) & (data[vals] < pvalue_threshold), "Significance"] = "Significant up"
    data.loc[(data["log2FoldChange"] < -fold_change_threshold) & (data[vals] < pvalue_threshold), "Significance"] = "Significant down"
    data["-log10(pvalue)"] = -np.log10(data[vals])
    colors = {"Significant up": "#EC0000", "Not Significant": "#DBDBDB","Significant down":"#0080F6","Significant":"#FCECCC"}
    data=data.reset_index(drop=True)
    draw_colors=[]
    for i in range(len(data)):
        draw_colors.append(colors[data['Significance'][i]])
    plt.scatter(data["log2FoldChange"], data["-log10(pvalue)"], c=draw_colors, alpha=0.5,edgecolors='none')
    # Add horizontal and vertical lines for thresholds
    plt.axhline(-np.log10(pvalue_threshold), color="blue", linestyle="--", linewidth=1, label="p-value threshold")
    plt.axvline(-fold_change_threshold, color="green", linestyle="--", linewidth=1, label="log2FC threshold")
    plt.axvline(fold_change_threshold, color="green", linestyle="--", linewidth=1)
    plt.xlim(-xlim,xlim)
    plt.ylim(0,ylim)
    for label, color in colors.items():
        plt.scatter([], [], color=color, label=label)
    # Titles and labels
    plt.title("Volcano Plot")
    plt.xlabel("log2(Fold Change)")
    plt.ylabel("-log10(p-value)")
    plt.legend(frameon=False, loc="upper right",bbox_to_anchor=(1.7, 1))
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    text=pd.read_table(text,header=None)
    text.columns=[gene_col]
    label_data=pd.merge(data,text,how='inner',on=[gene_col])
    plt.grid(alpha=0)
    plt.tight_layout()
    texts = []
    for _, row in label_data.iterrows():
        texts.append(plt.text(
        row['log2FoldChange'],
        row['-log10(pvalue)'],
        row[gene_col],
        fontsize=8,
        color='black',
        ha='center',
        va='center'))
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle='-', color='black', lw=0.5),  # 添加连接线
        expand_text=(1.2, 1.2),  # 扩展文本间距
        expand_points=(1.2, 1.2),  # 扩展点周围区域
        force_text=0.5,  # 调整力度
        force_points=0.5,
        lim=100)
    return plt
    
count_file = 'Melenoma_Count2.txt'
meta_file = 'Melenoma_Obs2.txt'
group1 = 'R' 
group2 = 'N' 
vals = "pvalue"
fold_change_threshold = 1 
pvalue_threshold = 0.05
xlim=5
ylim=20
factors=['response_NR','dataset_id']
text='genelist2.txt'
dfo = deseq2(count_file, meta_file, group1, group2,factors)
dfo.to_csv('Volcano_table2.xls',sep='\t',index=False)
plt=vocalno(dfo,fold_change_threshold, pvalue_threshold,vals,text,'Geneid',xlim,ylim)
plt.savefig('TestVolcanoTag2.pdf')




import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata


df1=pd.read_table('Melanoma-GSE78220_data.Response.tsv',sep='\t',index_col=0)
df2=pd.read_table('Melanoma-GSE91061_data.Response.tsv',sep='\t',index_col=0)
df3=pd.read_table('Melanoma-phs000452.Response_data.tsv',sep='\t',index_col=0)
df4=pd.read_table('Melanoma-PRJEB23709.Response_data.tsv',sep='\t',index_col=0)

df1=df1.set_index('GENE_SYMBOL')
df2=df2.set_index('GENE_SYMBOL')
df3=df3.set_index('GENE_SYMBOL')
df4=df4.set_index('GENE_SYMBOL')

adata1=ad.AnnData(X=df1.T)
adata2=ad.AnnData(X=df2.T)
adata3=ad.AnnData(X=df3.T)
adata4=ad.AnnData(X=df4.T)

adata1=QcNorm(adata1)
adata2=QcNorm(adata2)
adata3=QcNorm(adata3)
adata4=QcNorm(adata4)

adata=ad.concat([adata1,adata2,adata3,adata4])

df1o=pd.read_table('Melanoma-GSE78220.Response.tsv',sep='\t')
df2o=pd.read_table('Melanoma-GSE91061.Response.tsv',sep='\t')
df3o=pd.read_table('Melanoma-phs000452.Response.tsv',sep='\t')
df4o=pd.read_table('Melanoma-PRJEB23709.Response.tsv',sep='\t')
dfo=pd.DataFrame({'sample_id':list(adata.obs.index)})
tmp=pd.concat([df1o,df2o,df3o,df4o])
dfo=pd.merge(dfo,tmp,how='left',on=['sample_id'])
adata.obs=dfo
adata=adata[adata.obs['Therapy']=='anti-PD-1',:]

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

def Shap(model,X_test,adata,name):
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


adata.obs['judge']=list(adata.obs['response_NR'])
adata=adata[adata.obs['judge']!='UNK',:]

df=pd.read_table('/home/zhaoyue/project/Melenoma/data/Volcano_table.xls')
dfs=df[df['padj']<1]
genelist=list(dfs['Geneid'])
lists=[]
for i in list(adata.var.index):
    if i in genelist:
        lists.append(1)
    else:
        lists.append(0)
adata.var['pick']=lists
tmp=adata[:,adata.var['pick']==1]
#tmp=tmp[tmp.obs['Treatment']=='PRE',:]
tmp=tmp[tmp.obs['dataset_id']!='phs000452',:]
tmp=tmp[tmp.obs['dataset_id']!='PRJEB23709',:]
exp=tmp.X
label=list(tmp.obs['response_NR'])
label=[]
for i in list(tmp.obs['response_NR']):
    if i=='N':
        label.append(-1)
    else:
        label.append(1)
tmp.obs['label']=label
dfo=pd.read_table('genelist.txt',sep='\t')

genelists=[['BCAN','ALDOA','PLCG2','GABARAP','PRAME','PYGL','SPP1','GAGE2A','TYRP1','HSP90AB1','SERPINB6','BST2','MX2','ATP6V0C','COL1A2','AEBP1','RAB38','ALDH1A1','PFN1','NME1'],['IFI44L','GNLY','RAB12','ISG15','ZNF302','MYL12A','PSMB10','PLCG2','IFITM3','JAK3','KLF6','ACTB','S100A10','IFI6','PARK7','LCMT2','TBC1D17','KMT2E','FYB1','ITITM2'],['BCAN','PYGL','AEBP1','PRAME','PLCG2','ALDOA','BST2','TYRP1','KRT18','ATP6V0C','COL1A2','SRAF','GNLY','IFI44L','GAGE2A','XIST','HSP90AB1','CTSD','APLP2','SCRN1']]

#genelists=[['CA8','PTGDS','EZR','BAAT','FTH1','PRNP','TM4SF1','PYGL','PDE4DIP','CD44','MEGEA3','AHNAK2','VAV3','LAMP2','FXYD5','CSPG4','S100B','RPB34','IFITM3','RPL34'],['GNLY','RPS26','IFITM3','IFI44L','ZNF302','MT-CO1','ERAP2','RPL36AL','PLCG2','CD69','CNOT2','RAB12','FTH1','LCMT2','FYB1','TSC22D3','DIP2A','ZNF277','PPIL6','MYL12A'],['RPS26','GNLY','CA8','FTH1','IFI44L','PLCG2','IFITM3','ERAP2','FYB1','PTGDS','RPL36AL','SRAF','ZNF302','SRGN','MT-ND5','BAAT','CD69','RAB34','PPIL6','LINC01873']]

tplist=['malignant','T cells','All']
clist=['#F39900','#007900','#7B0000']
x=0
for genelist in genelists:
    adatas=tmp[:, tmp.var_names.isin(genelist)]
    print('data num',len(adatas.var))
    exp=adatas.X
    label=list(adatas.obs['label'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=clist[x],lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'{}'.format(tplist[x]))
    x+=1
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('AUC_bulk_cross_top.pdf'.format(cellname))




"""
df=pd.read_table('/home/dj/ZhaoYue_Home/project/Melenoma/data/Volcano_table.xls')
dfs=df[df['padj']<0.01]
genelist1=list(dfs['Geneid'])
df=pd.read_table('/home/dj/ZhaoYue_Home/project/Melenoma/data/Volcano_table2.xls')
dfs=df[df['pvalue']<0.01]
genelist2=list(dfs['Geneid'])
genelist=list(set(genelist1))+list(set(genelist2))
#genelist=genelist2
lists=[]
for i in list(adata.var.index):
    if i in genelist:
        lists.append(1)
    else:
        lists.append(0)
adata.var['pick']=lists
tmp=adata[:,adata.var['pick']==1]
exp=tmp.X
label=list(tmp.obs['response_NR'])
accuracy2,model3,X_test2,y_pred2=TumorClassify(exp,label)
print(accuracy2)



df=pd.read_table('/home/dj/ZhaoYue_Home/project/Melenoma/data/Volcano_table.xls')
dfs=df[df['padj']<0.01]
genelist1=list(dfs['Geneid'])
df=pd.read_table('/home/dj/ZhaoYue_Home/project/Melenoma/data/Volcano_table2.xls')
dfs=df[df['pvalue']<0.01]
genelist2=list(dfs['Geneid'])
genelist=list(set(genelist1))+list(set(genelist2))
#genelist=genelist2
lists=[]
for i in list(adata.var.index):
    if i in genelist:
        lists.append(1)
    else:
        lists.append(0)

import harmonypy as hm

adata.var['pick']=lists
tmp=tmp[tmp.obs['dataset_id']!='GSE78220',:]
tmp=tmp[tmp.obs['dataset_id']!='GSE91061',:]
batch_key = 'dataset_id'  # Replace with the column containing batch labels
sc.tl.pca(adata, svd_solver='arpack')  # Generates PCA-related attributes in adata
original_pcs = adata.obsm['X_pca']     # Extract the PC matrix (samples × PCs)
pca_loadings = adata.varm['PCs']       # Principal component loadings (genes × PCs)
pca_mean = adata.var['means'] if 'means' in adata.var else np.mean(adata.X, axis=0)  
corrected_pcs = hm.run_harmony(original_pcs, adata.obs, batch_key).Z_corr.T
harmonized_matrix = np.dot(corrected_pcs, pca_loadings.T) + pca_mean  # (N × M)
tmp=tmp[:,tmp.var['pick']==1]

exp=tmp.X
label=list(tmp.obs['response_NR'])
accuracy2,model3,X_test2,y_pred2=TumorClassify(exp,label)
print(accuracy2)
"""
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata


df1=pd.read_table('Melanoma-GSE78220_data.Response.tsv',sep='\t',index_col=0)
df2=pd.read_table('Melanoma-GSE91061_data.Response.tsv',sep='\t',index_col=0)
df3=pd.read_table('Melanoma-phs000452.Response_data.tsv',sep='\t',index_col=0)
df4=pd.read_table('Melanoma-PRJEB23709.Response_data.tsv',sep='\t',index_col=0)

df1=df1.set_index('GENE_SYMBOL')
df2=df2.set_index('GENE_SYMBOL')
df3=df3.set_index('GENE_SYMBOL')
df4=df4.set_index('GENE_SYMBOL')

adata1=ad.AnnData(X=df1.T)
adata2=ad.AnnData(X=df2.T)
adata3=ad.AnnData(X=df3.T)
adata4=ad.AnnData(X=df4.T)

adata1=QcNorm(adata1)
adata2=QcNorm(adata2)
adata3=QcNorm(adata3)
adata4=QcNorm(adata4)

adata=ad.concat([adata1,adata2,adata3,adata4])

df1o=pd.read_table('Melanoma-GSE78220.Response.tsv',sep='\t')
df2o=pd.read_table('Melanoma-GSE91061.Response.tsv',sep='\t')
df3o=pd.read_table('Melanoma-phs000452.Response.tsv',sep='\t')
df4o=pd.read_table('Melanoma-PRJEB23709.Response.tsv',sep='\t')
dfo=pd.DataFrame({'sample_id':list(adata.obs.index)})
tmp=pd.concat([df1o,df2o,df3o,df4o])
dfo=pd.merge(dfo,tmp,how='left',on=['sample_id'])
adata.obs=dfo
adata=adata[adata.obs['Therapy']=='anti-PD-1',:]

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

def Shap(model,X_test,adata,name):
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


adata.obs['judge']=list(adata.obs['response_NR'])
adata=adata[adata.obs['judge']!='UNK',:]

df=pd.read_table('/home/zhaoyue/project/Melenoma/data/Volcano_table.xls')
dfs=df[df['padj']<0.01]
genelist=list(dfs['Geneid'])
lists=[]
for i in list(adata.var.index):
    if i in genelist:
        lists.append(1)
    else:
        lists.append(0)
adata.var['pick']=lists
tmp=adata[:,adata.var['pick']==1]
#tmp=tmp[tmp.obs['Treatment']=='PRE',:]
tmp=tmp[tmp.obs['dataset_id']!='phs000452',:]
tmp=tmp[tmp.obs['dataset_id']!='PRJEB23709',:]
exp=tmp.X
label=list(tmp.obs['response_NR'])
label=[]
for i in list(tmp.obs['response_NR']):
    if i=='N':
        label.append(-1)
    else:
        label.append(1)
tmp.obs['label']=label
dfo=pd.read_table('genelist.txt',sep='\t')

toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
x=0
for j in toplist:
    dfx=dfo
    dfx=dfx.reset_index(drop=True)
    dfx=dfx.head(j)
    genelist=list(dfx['Feature'])
    adatas=tmp[:, tmp.var_names.isin(genelist)]
    exp=adatas.X
    label=list(adatas.obs['label'])
    print(set(label))
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')
    #dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=clist[x],lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top{}'.format(j))
    x+=1
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('AUC_bulk_top.pdf'.format(cellname))




"""
df=pd.read_table('/home/dj/ZhaoYue_Home/project/Melenoma/data/Volcano_table.xls')
dfs=df[df['padj']<0.01]
genelist1=list(dfs['Geneid'])
df=pd.read_table('/home/dj/ZhaoYue_Home/project/Melenoma/data/Volcano_table2.xls')
dfs=df[df['pvalue']<0.01]
genelist2=list(dfs['Geneid'])
genelist=list(set(genelist1))+list(set(genelist2))
#genelist=genelist2
lists=[]
for i in list(adata.var.index):
    if i in genelist:
        lists.append(1)
    else:
        lists.append(0)
adata.var['pick']=lists
tmp=adata[:,adata.var['pick']==1]
exp=tmp.X
label=list(tmp.obs['response_NR'])
accuracy2,model3,X_test2,y_pred2=TumorClassify(exp,label)
print(accuracy2)



df=pd.read_table('/home/dj/ZhaoYue_Home/project/Melenoma/data/Volcano_table.xls')
dfs=df[df['padj']<0.01]
genelist1=list(dfs['Geneid'])
df=pd.read_table('/home/dj/ZhaoYue_Home/project/Melenoma/data/Volcano_table2.xls')
dfs=df[df['pvalue']<0.01]
genelist2=list(dfs['Geneid'])
genelist=list(set(genelist1))+list(set(genelist2))
#genelist=genelist2
lists=[]
for i in list(adata.var.index):
    if i in genelist:
        lists.append(1)
    else:
        lists.append(0)

import harmonypy as hm

adata.var['pick']=lists
tmp=tmp[tmp.obs['dataset_id']!='GSE78220',:]
tmp=tmp[tmp.obs['dataset_id']!='GSE91061',:]
batch_key = 'dataset_id'  # Replace with the column containing batch labels
sc.tl.pca(adata, svd_solver='arpack')  # Generates PCA-related attributes in adata
original_pcs = adata.obsm['X_pca']     # Extract the PC matrix (samples × PCs)
pca_loadings = adata.varm['PCs']       # Principal component loadings (genes × PCs)
pca_mean = adata.var['means'] if 'means' in adata.var else np.mean(adata.X, axis=0)  
corrected_pcs = hm.run_harmony(original_pcs, adata.obs, batch_key).Z_corr.T
harmonized_matrix = np.dot(corrected_pcs, pca_loadings.T) + pca_mean  # (N × M)
tmp=tmp[:,tmp.var['pick']==1]

exp=tmp.X
label=list(tmp.obs['response_NR'])
accuracy2,model3,X_test2,y_pred2=TumorClassify(exp,label)
print(accuracy2)
"""
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset


def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata


df1=pd.read_table('Melanoma-GSE78220_data.Response.tsv',sep='\t',index_col=0)
df2=pd.read_table('Melanoma-GSE91061_data.Response.tsv',sep='\t',index_col=0)
df3=pd.read_table('Melanoma-phs000452.Response_data.tsv',sep='\t',index_col=0)
df4=pd.read_table('Melanoma-PRJEB23709.Response_data.tsv',sep='\t',index_col=0)

df1=df1.set_index('GENE_SYMBOL')
df2=df2.set_index('GENE_SYMBOL')
df3=df3.set_index('GENE_SYMBOL')
df4=df4.set_index('GENE_SYMBOL')

adata1=ad.AnnData(X=df1.T)
adata2=ad.AnnData(X=df2.T)
adata3=ad.AnnData(X=df3.T)
adata4=ad.AnnData(X=df4.T)

#adata1=QcNorm(adata1)
#adata2=QcNorm(adata2)
#adata3=QcNorm(adata3)
#adata4=QcNorm(adata4)

adata=ad.concat([adata1,adata2,adata3,adata4])

df1o=pd.read_table('Melanoma-GSE78220.Response.tsv',sep='\t')
df2o=pd.read_table('Melanoma-GSE91061.Response.tsv',sep='\t')
df3o=pd.read_table('Melanoma-phs000452.Response.tsv',sep='\t')
df4o=pd.read_table('Melanoma-PRJEB23709.Response.tsv',sep='\t')
dfo=pd.DataFrame({'sample_id':list(adata.obs.index)})
tmp=pd.concat([df1o,df2o,df3o,df4o])
dfo=pd.merge(dfo,tmp,how='left',on=['sample_id'])
adata.obs=dfo
adata=adata[adata.obs['Therapy']=='anti-PD-1',:]
adata.obs['judge']=list(adata.obs['response_NR'])
adata=adata[adata.obs['judge']!='UNK',:]
adata.write_h5ad('ICB_Bulk.h5ad')

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

def Shap(model,X_test,adata,name):
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


df=pd.read_table('/home/dj/ZhaoYue_Home/project/Melenoma/data/Volcano_table.xls')
dfs=df[df['padj']<0.01]
genelist=list(dfs['Geneid'])
lists=[]
for i in list(adata.var.index):
    if i in genelist:
        lists.append(1)
    else:
        lists.append(0)
adata.var['pick']=lists
tmp=adata[:,adata.var['pick']==1]
#tmp=tmp[tmp.obs['Treatment']=='PRE',:]
tmp=tmp[tmp.obs['dataset_id']!='phs000452',:]
tmp=tmp[tmp.obs['dataset_id']!='PRJEB23709',:]
exp=tmp.X
label=list(tmp.obs['response_NR'])
accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
print(accuracy2)
dfs=Shap(model3,X_test2,tmp,'pre')
dfs.to_csv('genelist.txt',sep='\t',index=False)
from sklearn.metrics import roc_curve, auc


exp=tmp.X
label=[]
for i in list(tmp.obs['response_NR']):
    if i=='N':
        label.append(-1)
    else:
        label.append(1)
X_train, X_test, y_train, y_test = train_test_split(exp, label, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
dfx=pd.DataFrame({'data':y_test,'pred':y_pred})
x=0
for i in range(len(dfx)):
    if dfx['data'][i]==1 and dfx['pred'][i]>=0:
        x+=1
    elif  dfx['data'][i]==-1 and dfx['pred'][i]<0:
        x+=1
accuracy2=x/len(dfx)
print('regression',accuracy2)


df=pd.read_table('/home/dj/ZhaoYue_Home/project/Melenoma/data/Volcano_table2.xls')
dfs=df[df['pvalue']<0.01]
genelist=list(dfs['Geneid'])
lists=[]
for i in list(adata.var.index):
    if i in genelist:
        lists.append(1)
    else:
        lists.append(0)
adata.var['pick']=lists
tmp=adata[:,adata.var['pick']==1]
tmp=tmp[tmp.obs['dataset_id']!='GSE78220',:]
tmp=tmp[tmp.obs['dataset_id']!='GSE91061',:]
#tmp=tmp[tmp.obs['dataset_id']!='PRJEB23709',:]
exp=tmp.X
label=[]
for i in list(tmp.obs['response_NR']):
    if i=='N':
        label.append(-1)
    else:
        label.append(1)
X_train, X_test, y_train, y_test = train_test_split(exp, label, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
dfx=pd.DataFrame({'data':y_test,'pred':y_pred})
x=0
for i in range(len(dfx)):
    if dfx['data'][i]==1 and dfx['pred'][i]>=0:
        x+=1
    elif  dfx['data'][i]==-1 and dfx['pred'][i]<0:
        x+=1
accuracy2=x/len(dfx)
#accuracy2,model3,X_test2,y_pred2=TumorClassify(exp,label)
print(accuracy2)
dfs=Shap(model,X_test,tmp,y_pred,'pre')
dfs.to_csv('genelist2.txt',sep='\t',index=False)
y_test_binary = np.where(y_test == 1, 1, 0)  # Convert labels to 0 and 1
y_test_binary = np.array(y_test_binary).ravel()  # Flatten the array if needed
y_pred = np.array(y_pred).ravel()
y_test_binary=[]
for i in y_test:
    if i==1:
       y_test_binary.append(1)
    else:
       y_test_binary.append(0)
print(y_test_binary,y_pred)
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred)  # Use y_pred as the score
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')  # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('AUC_batch2.pdf')


"""
df=pd.read_table('/home/dj/ZhaoYue_Home/project/Melenoma/data/Volcano_table.xls')
dfs=df[df['padj']<0.01]
genelist1=list(dfs['Geneid'])
df=pd.read_table('/home/dj/ZhaoYue_Home/project/Melenoma/data/Volcano_table2.xls')
dfs=df[df['pvalue']<0.01]
genelist2=list(dfs['Geneid'])
genelist=list(set(genelist1))+list(set(genelist2))
#genelist=genelist2
lists=[]
for i in list(adata.var.index):
    if i in genelist:
        lists.append(1)
    else:
        lists.append(0)
adata.var['pick']=lists
tmp=adata[:,adata.var['pick']==1]
exp=tmp.X
label=list(tmp.obs['response_NR'])
accuracy2,model3,X_test2,y_pred2=TumorClassify(exp,label)
print(accuracy2)



df=pd.read_table('/home/dj/ZhaoYue_Home/project/Melenoma/data/Volcano_table.xls')
dfs=df[df['padj']<0.01]
genelist1=list(dfs['Geneid'])
df=pd.read_table('/home/dj/ZhaoYue_Home/project/Melenoma/data/Volcano_table2.xls')
dfs=df[df['pvalue']<0.01]
genelist2=list(dfs['Geneid'])
genelist=list(set(genelist1))+list(set(genelist2))
#genelist=genelist2
lists=[]
for i in list(adata.var.index):
    if i in genelist:
        lists.append(1)
    else:
        lists.append(0)

import harmonypy as hm

adata.var['pick']=lists
tmp=tmp[tmp.obs['dataset_id']!='GSE78220',:]
tmp=tmp[tmp.obs['dataset_id']!='GSE91061',:]
batch_key = 'dataset_id'  # Replace with the column containing batch labels
sc.tl.pca(adata, svd_solver='arpack')  # Generates PCA-related attributes in adata
original_pcs = adata.obsm['X_pca']     # Extract the PC matrix (samples × PCs)
pca_loadings = adata.varm['PCs']       # Principal component loadings (genes × PCs)
pca_mean = adata.var['means'] if 'means' in adata.var else np.mean(adata.X, axis=0)  
corrected_pcs = hm.run_harmony(original_pcs, adata.obs, batch_key).Z_corr.T
harmonized_matrix = np.dot(corrected_pcs, pca_loadings.T) + pca_mean  # (N × M)
tmp=tmp[:,tmp.var['pick']==1]

exp=tmp.X
label=list(tmp.obs['response_NR'])
accuracy2,model3,X_test2,y_pred2=TumorClassify(exp,label)
print(accuracy2)
"""
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import Dataset



df1=pd.read_table('GSE91061_BMS038109Sample.hg19KnownGene.raw.csv',sep=',',index_col=0)
df2=pd.read_table('GSE78220_raw_counts_GRCh38.p13_NCBI.tsv',sep='\t')
df2=df2.set_index('GeneID')


adata1=ad.AnnData(X=df1.T)
adata2=ad.AnnData(X=df2.T)
adata=ad.concat([adata1,adata2])

df1o=pd.read_table('Melanoma-GSE78220.Response.tsv',sep='\t')
df2o=pd.read_table('Melanoma-GSE91061.Response.tsv',sep='\t')
dfo=pd.DataFrame({'sample_id':list(adata.obs.index)})
tmp=pd.concat([df1o,df2o,])
dfo=pd.merge(dfo,tmp,how='left',on=['sample_id'])
adata.obs=dfo
adata=adata[adata.obs['Therapy']=='anti-PD-1',:]
lists=[]
for i in list(adata.obs['Treatment']):
    lists.append(i)
adata.obs['Treat']=lists
adata=adata[adata.obs['Treat']=='PRE',:]
adata=adata[adata.obs['dataset_id']!='phs000452',:]
adata=adata[adata.obs['dataset_id']!='PRJEB23709',:]
dfo=pd.DataFrame(adata.X)
dfo=dfo.T
dfo.columns=list(adata.obs['sample_id'])
dfo.index=list(adata.var.index)
dfo.to_csv('Melenoma_CountPre.txt',sep='\t')
adata.obs.to_csv('Melenoma_ObsPre.txt',sep='\t')






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

adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata.raw = adata.copy()
featurelist=['MFAP2','TRIP1','FOXC2','PLG','USP51','CKB','FLOT2','CNR2','UGT3A2','HS6ST1','GP5','SERPINC1','HTR6','SERPINB2','CYP3A4','SLC47A1','SLC30A2','CIDEB','CLYBL']
#featurelist=['FOXC2','HTR6', 'CEP295NL','STUM','PON3', 'LIPG', 'GAD1','ACOX2','PROC','TSPAN8','FCAMR','A1BG']

genelist=[]
for gene in featurelist:
    if gene in adata.var_names and gene in list(adata.var.index):
        count=count_cells_expressing_gene(adata, gene, threshold=0)
        print(gene,count)
        if count>100:
            genelist.append(gene)
n_genes = len(genelist)
n_cols = 5  # Number of columns
n_rows = (n_genes // n_cols) + (n_genes % n_cols > 0)  # Calculate rows needed
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = axes.flatten()


adata.var.index = adata.var.index.str.strip()
genelist = [gene.strip() for gene in genelist]

#genelist=['FOXC2','HTR6', 'CEP295NL','STUM','PON3', 'LIPG', 'GAD1','ACOX2','PROC','TSPAN8','FCAMR','A1BG']
n_genes = len(genelist)
n_cols = 5  # Number of columns
n_rows = (n_genes // n_cols) + (n_genes % n_cols > 0)  # Calculate rows needed
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = axes.flatten()
for x, gene in enumerate(genelist):
    sc.pl.umap(adata,color=gene,ax=axes[x],show=False,title=gene, color_map="viridis")
plt.savefig('MelanomaFeatures.pdf')

cells=['Malignant', 'T_cell']
df=pd.read_table('celltype_keygenes.xls',sep='\t')
genelists=[]
for i in cells:
    dfs=df[df['cluster']==i]
    dfs=dfs.head(20)
    lists=list(dfs['Feature'])
    genelists.append(lists)
df=pd.read_table('All_keygenes.xls',sep='\t')
df=df.head(20)
text=list(df['Feature'])
genelists.append(text)
print(genelists)
xlist=['Malignant', 'T_cell','All']
t=0
for genelist in genelists:
    n_genes = len(genelist)
    n_cols = 5  # Number of columns
    n_rows = (n_genes // n_cols) + (n_genes % n_cols > 0)  # Calculate rows needed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()
    for x, gene in enumerate(genelist):
        sc.pl.umap(adata,color=gene,ax=axes[x],show=False,title=gene, color_map="viridis")
    plt.savefig('Melanoma_{}_Features.pdf'.format(xlist[t]))
    t+=1






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
    print(X_train)
    return X_train, X_test, y_train, y_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
#adata=sc.read_h5ad('ICB.h5ad')
adata2=adata[adata.obs['pre_post']=='Pre',:]
lists=[]
for i in list(adata2.obs['Combined_outcome']):
    if i=='Favourable':
        lists.append('R')
    else:
        lists.append('N')
adata2.obs['response']=lists

dflist=[]
cells=['Malignant', 'T_cell']

adata=adata2.copy()
dfo=pd.read_table('celltype_keygenes.xls',sep='\t')
toplist=[20]
clist=['#F39900','#007900','#7B0000']
dflist=[]
for types in cells:
    adatac,lbs=SamplePick(adata,types,'author_cell_type_update','response')
    print(set(lbs))
    if lbs!=None:
        if len(lbs)>1:
            x=0 
            for j in toplist:
                cellname=types.replace(' ', '_')
                adatas=adatac.copy()
                dfx=dfo[dfo['cluster']==types]
                dfx=dfx.reset_index(drop=True)
                dfx=dfx.head(j)
                genelist=list(dfx['Feature'])
                adatas=adatas[:, adatas.var_names.isin(genelist)]
                exp=adatas.X
                label=list(adatas.obs['response'])
                X_train, X_test, y_train, y_test = TumorClassify(exp,label)
                print(set(y_train),set(y_test))
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
                model = KAN(width=[20,3,1, 1],grid=50,k=5,seed=42,device=device )
                model.fit(dataset, opt="LBFGS", steps=33, lamb=0.003)
                model.plot()
                plt.savefig('KAN_{}_top{}.pdf'.format(cellname,j))
                plt.close()
                logits = model(X_test_tensor)
                print('logits',logits)
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
                plt.savefig('KAN_AUC_{}_top{}.pdf'.format(cellname,j))  # Save the ROC curve
                plt.close()






import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata=adata[adata.obs['pre_post']=='Pre',:]
adata=adata[adata.obs['Combined_outcome']!='UT',:]
lists=[]
for i in list(adata.obs['Combined_outcome']):
    if i=='Favourable':
        lists.append('R')
    else:
        lists.append('N')
adata.obs['response']=lists
adata2=adata.copy()
dfo=pd.read_table('celltype_keygenes.xls',sep='\t')
cells=['Malignant', 'T_cell']
genelists=[['MFAP2','PLG','USP51','CKB','FLOT2','CNR2','UGT3A2','HS6ST1','GPS','SERPINB2','CYP3A4','SLC47A1','SLC30A2','CLYBL']]
clist=['#F39900','#007900','#7B0000']
for i in cells:
    plt.figure(figsize=(8, 6))
    x=0
    for genelist in genelists:
        adatas=adata2[adata2.obs['author_cell_type_update']==i,:]
        adatas=adatas[:, adatas.var_names.isin(genelist)]
        exp=adatas.X
        label=list(adatas.obs['response'])
        accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
        cellname=i.replace(' ', '_')
        dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=clist[x],lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'cross')
        x+=1
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig('AUC_{}_cross_top.pdf'.format(cellname))       

dfs=pd.read_table('All_keygenes.xls',sep='\t')
plt.figure(figsize=(8, 6))
x=0
for genelist in genelists:
    adatas=adata2[:, adata2.var_names.isin(genelist)]
    exp=adatas.X
    label=list(adatas.obs['response'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top')
    x+=1
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('AUC_All_cross_top.pdf'.format(cellname))


import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata=adata[adata.obs['pre_post']=='Pre',:]
adata=adata[adata.obs['Combined_outcome']!='UT',:]
lists=[]
for i in list(adata.obs['Combined_outcome']):
    if i=='Favourable':
        lists.append('R')
    else:
        lists.append('N')
adata.obs['response']=lists
adata2=adata.copy()
dfo=pd.read_table('celltype_keygenes.xls',sep='\t')
cells=['Malignant', 'T_cell']
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
for i in cells:
    plt.figure(figsize=(8, 6))
    x=0
    for j in toplist:
        dfx=dfo[dfo['cluster']==i]
        dfx=dfx.reset_index(drop=True)
        dfx=dfx.head(j)
        genelist=list(dfx['Feature']) 
        adatas=adata2[adata2.obs['author_cell_type_update']==i,:]
        adatas=adatas[:, adatas.var_names.isin(genelist)]
        exp=adatas.X
        label=list(adatas.obs['response'])
        accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
        cellname=i.replace(' ', '_')
        dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=clist[x],lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top{}'.format(j))
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

dfs=pd.read_table('All_keygenes.xls',sep='\t')
plt.figure(figsize=(8, 6))
x=0
for j in toplist:
    dfx=dfo[dfo['cluster']==i]
    dfx=dfx.reset_index(drop=True)
    dfx=dfx.head(j)
    genelist=list(dfx['Feature'])
    adatas=adata2[:, adata2.var_names.isin(genelist)]
    exp=adatas.X
    label=list(adatas.obs['response'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top{}'.format(j))
    x+=1
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('AUC_All_top.pdf'.format(cellname))


import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

choselist=['Metastatic','Primary']
adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata=adata[adata.obs['Primary_or_met'].isin(choselist),:]
adata=adata[adata.obs['Combined_outcome']=='UT',:]
adata2=adata.copy()
dfo=pd.read_table('celltype_Primary_or_met_keygenes.xls',sep='\t')
cells=['Malignant', 'T_cell']
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
for i in cells:
    plt.figure(figsize=(8, 6))
    x=0
    for j in toplist:
        dfx=dfo[dfo['cluster']==i]
        dfx=dfx.reset_index(drop=True)
        dfx=dfx.head(j)
        genelist=list(dfx['Feature']) 
        adatas=adata2[adata2.obs['author_cell_type_update']==i,:]
        adatas=adatas[:, adatas.var_names.isin(genelist)]
        exp=adatas.X
        label=list(adatas.obs['Primary_or_met'])
        accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
        cellname=i.replace(' ', '_')
        dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=clist[x],lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top{}'.format(j))
        x+=1
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig('AUC_{}_Primary_or_met_top.pdf'.format(cellname))       

dfs=pd.read_table('All_Primary_or_met_keygenes.xls',sep='\t')
plt.figure(figsize=(8, 6))
x=0
for j in toplist:
    dfx=dfo[dfo['cluster']==i]
    dfx=dfx.reset_index(drop=True)
    dfx=dfx.head(j)
    genelist=list(dfx['Feature'])
    adatas=adata2[:, adata2.var_names.isin(genelist)]
    exp=adatas.X
    label=list(adatas.obs['Primary_or_met'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top{}'.format(j))
    x+=1
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('AUC_All_Primary_or_met_top.pdf'.format(cellname))


import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

choselist=['Metastatic','Primary']
adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata=adata[adata.obs['Primary_or_met'].isin(choselist),:]
adata=adata[adata.obs['Combined_outcome']=='UT',:]
adata2=adata.copy()
dflist=[]
cells=['Malignant', 'T_cell']
for i in cells:
    adatas=adata2[adata2.obs['author_cell_type_update']==i,:]
    exp=adatas.X
    label=list(adatas.obs['Primary_or_met'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')+'_Primary_or_met_'
    dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
    xtmp=dfs.head(20)
    xtmp=xtmp.loc[:,['Feature']]
    xtmp.to_csv('genelist_{}.txt'.format(i),sep='\t',index=False,header=False)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
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
    dfs['cluster']=[i]*len(dfs)
    dflist.append(dfs)
    plt.savefig('AUC_Primary_or_met_{}.pdf'.format(cellname))
dfo=pd.concat(dflist)
dfo.to_csv('celltype_Primary_or_met_keygenes.xls',sep='\t',index=False)

exp=adata2.X
label=list(adata2.obs['Primary_or_met'])
accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
cellname=i.replace(' ', '_')
dfs=Shap(model3,X_test2,adata2,y_pred_label,label,'All_Primary_or_met')
xtmp=dfs.head(20)
xtmp=xtmp.loc[:,['Feature']]
xtmp.to_csv('genelist_all.txt',sep='\t',index=False,header=False)
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
plt.savefig('AUC_Primary_or_met_All.pdf'.format(cellname))
dfs.to_csv('All_Primary_or_met_keygenes.xls',sep='\t',index=False)

import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 
import scrublet as scr
import harmonypy as hm

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

mt_tag='MT-'
output='Melanoma'

adata=sc.read_h5ad('PanCancerICB.h5ad')
xlist=[]
for i in list(adata.var['feature_name']):
    i=i.split('_')
    xlist.append(i[0])
adata.var['ensembl']=list(adata.var.index)
adata.var.index=xlist
adata=adata[adata.obs['disease'].str.contains('melanoma')==True,:]


adata.var['mt'] = adata.var_names.str.startswith(mt_tag)  # For human datasets
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata = adata[adata.obs['pct_counts_mt'] < 20, :]
sc.pl.violin(adata, ['pct_counts_mt'], jitter=0.4, groupby='donor_id')
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
sc.pl.violin(adata, keys=['total_counts'], jitter=0.4, groupby='donor_id', rotation=45)
plt.savefig('{}_counts_violin.pdf'.format(output))
plt.close()
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts',color='donor_id', title="Total Counts vs Number of Genes")
plt.savefig('{}_counts_scatter.pdf'.format(output))
plt.close()

#adata = adata[(adata.obs['n_genes_by_counts'] > 200) & (adata.obs['n_genes_by_counts'] < 2500), :]
#Ribosome genes
adata.var['ribosomal'] = adata.var_names.str.startswith(('RPL', 'RPS'))
# Calculate percentage of ribosomal RNA
sc.pp.calculate_qc_metrics(adata, qc_vars=['ribosomal'], percent_top=None, log1p=False, inplace=True)
# Visualize ribosomal RNA content
sc.pl.violin(adata, keys=['pct_counts_ribosomal'], jitter=0.4, groupby=None, rotation=45)
plt.savefig('{}_ribosome_violin.pdf'.format(output))
plt.close()


pattern = re.compile(r'^(MT-|RPL|RPS)')
genes_to_remove = adata.var_names[
    adata.var_names.str.match(pattern)]
adata = adata[:, ~adata.var_names.isin(genes_to_remove)]

sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=3000)

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
adata.X.data[np.isnan(adata.X.data)] = 0 
sc.pp.log1p(adata)  # Log-transform the data
sc.pp.scale(adata, max_value=10)

sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, key_added='leiden')
fig = plt.figure(figsize=(20,15))
ax1 = fig.add_axes([0.05,0.5,0.3,0.3])
ax2 = fig.add_axes([0.5,0.5,0.3,0.3])
sc.pl.umap(adata, color='donor_id', ax=ax1,
               show=False, title="Before Batch Effect Removal")
sc.pl.umap(adata, color='leiden', ax=ax2,
               show=False, title="Before Batch Effect Removal")
reduced_data = adata.obsm['X_pca']
metadata = adata.obs[['donor_id']].astype(str).copy()
harmony_out = hm.run_harmony(
        data_mat = reduced_data,
        meta_data = metadata,
        vars_use = ['donor_id'],
        max_iter_harmony = 20,
        epsilon_harmony = 1e-4)
adata.obsm['X_pca_harmony'] = harmony_out.Z_corr.T
sc.pp.neighbors(adata, use_rep='X_pca_harmony')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_har')
ax3 = fig.add_axes([0.05,0.1,0.3,0.3])
ax4 = fig.add_axes([0.5,0.1,0.3,0.3])
sc.pl.umap(adata, color='donor_id', ax=ax3,
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
ax3.legend([], [], frameon=False)
plt.savefig('{}_UMAP.pdf'.format(output))

fig = plt.figure(figsize=(26,7))
axes = fig.add_axes([0.1,0.2,0.25,0.65])
sc.pl.umap(adata, color='cell_type', ax=axes,
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
plt.savefig('{}_CellType.pdf'.format(output))
plt.close()

for col in adata.var.select_dtypes(['category']).columns:
    adata.var[col] = adata.var[col].astype(str)
adata.write_h5ad('Melanoma_ICB_scRNA.h5ad')




















import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata=adata[adata.obs['pre_post']=='Post',:]
adata=adata[adata.obs['tissue']=='brain',:]
adata=adata[adata.obs['Combined_outcome']!='UT',:]
lists=[]
for i in list(adata.obs['Combined_outcome']):
    if i=='Favourable':
        lists.append('R')
    else:
        lists.append('N')
adata.obs['response']=lists
adata2=adata.copy()
dfo=pd.read_table('celltype_PostResponse_brain_keygenes.xls',sep='\t')
cells=['Malignant', 'T_cell']
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
for i in cells:
    plt.figure(figsize=(8, 6))
    x=0
    for j in toplist:
        dfx=dfo[dfo['cluster']==i]
        dfx=dfx.reset_index(drop=True)
        dfx=dfx.head(j)
        genelist=list(dfx['Feature']) 
        adatas=adata2[adata2.obs['author_cell_type_update']==i,:]
        adatas=adatas[:, adatas.var_names.isin(genelist)]
        exp=adatas.X
        label=list(adatas.obs['response'])
        accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
        cellname=i.replace(' ', '_')
        dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=clist[x],lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top{}'.format(j))
        x+=1
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig('AUC_{}_PostResponse_brain_top.pdf'.format(cellname))       

dfs=pd.read_table('All_PostResponse_brain_keygenes.xls',sep='\t')
plt.figure(figsize=(8, 6))
x=0
for j in toplist:
    dfx=dfo[dfo['cluster']==i]
    dfx=dfx.reset_index(drop=True)
    dfx=dfx.head(j)
    genelist=list(dfx['Feature'])
    adatas=adata2[:, adata2.var_names.isin(genelist)]
    exp=adatas.X
    label=list(adatas.obs['response'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top{}'.format(j))
    x+=1
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('AUC_All_PostResponse_brain_top.pdf'.format(cellname))


import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata=adata[adata.obs['pre_post']=='Post',:]
adata=adata[adata.obs['tissue']=='brain',:]
adata=adata[adata.obs['Combined_outcome']!='UT',:]
lists=[]
for i in list(adata.obs['Combined_outcome']):
    if i=='Favourable':
        lists.append('R')
    else:
        lists.append('N')
adata.obs['response']=lists
adata2=adata.copy()
dflist=[]
cells=['Malignant', 'T_cell']
for i in cells:
    adatas=adata2[adata2.obs['author_cell_type_update']==i,:]
    exp=adatas.X
    label=list(adatas.obs['response'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')+'_PostResponse_brain_'
    dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
    xtmp=dfs.head(20)
    xtmp=xtmp.loc[:,['Feature']]
    xtmp.to_csv('genelist_{}.txt'.format(i),sep='\t',index=False,header=False)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
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
    dfs['cluster']=[i]*len(dfs)
    dflist.append(dfs)
    plt.savefig('AUC_PostResponse_brain_{}.pdf'.format(cellname))
dfo=pd.concat(dflist)
dfo.to_csv('celltype_PostResponse_brain_keygenes.xls',sep='\t',index=False)

exp=adata2.X
label=list(adata2.obs['response'])
accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
cellname=i.replace(' ', '_')
dfs=Shap(model3,X_test2,adata2,y_pred_label,label,'All_PostResponse_brain')
xtmp=dfs.head(20)
xtmp=xtmp.loc[:,['Feature']]
xtmp.to_csv('genelist_all.txt',sep='\t',index=False,header=False)
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
plt.savefig('AUC_PostResponse_brain_All.pdf'.format(cellname))
dfs.to_csv('All_PostResponse_brain_keygenes.xls',sep='\t',index=False)

import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata=adata[adata.obs['pre_post']=='Post',:]
adata=adata[adata.obs['Combined_outcome']!='UT',:]
lists=[]
for i in list(adata.obs['Combined_outcome']):
    if i=='Favourable':
        lists.append('R')
    else:
        lists.append('N')
adata.obs['response']=lists
adata2=adata.copy()
dfo=pd.read_table('celltype_PostResponse_keygenes.xls',sep='\t')
cells=['Malignant', 'T_cell']
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
for i in cells:
    plt.figure(figsize=(8, 6))
    x=0
    for j in toplist:
        dfx=dfo[dfo['cluster']==i]
        dfx=dfx.reset_index(drop=True)
        dfx=dfx.head(j)
        genelist=list(dfx['Feature']) 
        adatas=adata2[adata2.obs['author_cell_type_update']==i,:]
        adatas=adatas[:, adatas.var_names.isin(genelist)]
        exp=adatas.X
        label=list(adatas.obs['response'])
        accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
        cellname=i.replace(' ', '_')
        dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=clist[x],lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top{}'.format(j))
        x+=1
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig('AUC_{}_PostResponse_top.pdf'.format(cellname))       

dfs=pd.read_table('All_PostResponse_keygenes.xls',sep='\t')
plt.figure(figsize=(8, 6))
x=0
for j in toplist:
    dfx=dfo[dfo['cluster']==i]
    dfx=dfx.reset_index(drop=True)
    dfx=dfx.head(j)
    genelist=list(dfx['Feature'])
    adatas=adata2[:, adata2.var_names.isin(genelist)]
    exp=adatas.X
    label=list(adatas.obs['response'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top{}'.format(j))
    x+=1
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('AUC_All_PostResponse_top.pdf'.format(cellname))


import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata=adata[adata.obs['pre_post']=='Post',:]
adata=adata[adata.obs['Combined_outcome']!='UT',:]
lists=[]
for i in list(adata.obs['Combined_outcome']):
    if i=='Favourable':
        lists.append('R')
    else:
        lists.append('N')
adata.obs['response']=lists
adata2=adata.copy()
dflist=[]
cells=['Malignant', 'T_cell']
for i in cells:
    adatas=adata2[adata2.obs['author_cell_type_update']==i,:]
    exp=adatas.X
    label=list(adatas.obs['response'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')+'_PostResponse_'
    dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
    xtmp=dfs.head(20)
    xtmp=xtmp.loc[:,['Feature']]
    xtmp.to_csv('genelist_{}.txt'.format(i),sep='\t',index=False,header=False)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
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
    dfs['cluster']=[i]*len(dfs)
    dflist.append(dfs)
    plt.savefig('AUC_PostResponse_{}.pdf'.format(cellname))
dfo=pd.concat(dflist)
dfo.to_csv('celltype_PostResponse_keygenes.xls',sep='\t',index=False)

exp=adata2.X
label=list(adata2.obs['response'])
accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
cellname=i.replace(' ', '_')
dfs=Shap(model3,X_test2,adata2,y_pred_label,label,'All_PostResponse')
xtmp=dfs.head(20)
xtmp=xtmp.loc[:,['Feature']]
xtmp.to_csv('genelist_all.txt',sep='\t',index=False,header=False)
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
plt.savefig('AUC_PostResponse_All.pdf'.format(cellname))
dfs.to_csv('All_PostResponse_keygenes.xls',sep='\t',index=False)

import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata=adata[adata.obs['pre_post']=='Pre',:]
adata=adata[adata.obs['Combined_outcome']!='UT',:]
lists=[]
for i in list(adata.obs['Combined_outcome']):
    if i=='Favourable':
        lists.append('R')
    else:
        lists.append('N')
adata.obs['response']=lists
adata2=adata.copy()
dfo=pd.read_table('celltype_PrePost_keygenes.xls',sep='\t')
cells=['Malignant', 'T_cell']
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
for i in cells:
    plt.figure(figsize=(8, 6))
    x=0
    for j in toplist:
        dfx=dfo[dfo['cluster']==i]
        dfx=dfx.reset_index(drop=True)
        dfx=dfx.head(j)
        genelist=list(dfx['Feature']) 
        adatas=adata2[adata2.obs['author_cell_type_update']==i,:]
        adatas=adatas[:, adatas.var_names.isin(genelist)]
        exp=adatas.X
        label=list(adatas.obs['response'])
        accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
        cellname=i.replace(' ', '_')
        dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=clist[x],lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top{}'.format(j))
        x+=1
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig('AUC_{}_PrePost_top.pdf'.format(cellname))       

dfs=pd.read_table('All_PrePost_keygenes.xls',sep='\t')
plt.figure(figsize=(8, 6))
x=0
for j in toplist:
    dfx=dfo[dfo['cluster']==i]
    dfx=dfx.reset_index(drop=True)
    dfx=dfx.head(j)
    genelist=list(dfx['Feature'])
    adatas=adata2[:, adata2.var_names.isin(genelist)]
    exp=adatas.X
    label=list(adatas.obs['response'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top{}'.format(j))
    x+=1
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('AUC_All_PrePost_top.pdf'.format(cellname))


import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata2=adata.copy()
dflist=[]
cells=['Malignant', 'T_cell']
for i in cells:
    adatas=adata2[adata2.obs['author_cell_type_update']==i,:]
    exp=adatas.X
    label=list(adatas.obs['pre_post'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')
    dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
    xtmp=dfs.head(20)
    xtmp=xtmp.loc[:,['Feature']]
    xtmp.to_csv('genelist_{}.txt'.format(i),sep='\t',index=False,header=False)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
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
    dfs['cluster']=[i]*len(dfs)
    dflist.append(dfs)
    plt.savefig('AUC_PrePost_{}.pdf'.format(cellname))
dfo=pd.concat(dflist)
dfo.to_csv('celltype_PrePost_keygenes.xls',sep='\t',index=False)

exp=adata2.X
label=list(adata2.obs['pre_post'])
accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
cellname=i.replace(' ', '_')
dfs=Shap(model3,X_test2,adata2,y_pred_label,label,'All')
xtmp=dfs.head(20)
xtmp=xtmp.loc[:,['Feature']]
xtmp.to_csv('genelist_all.txt',sep='\t',index=False,header=False)
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
plt.savefig('AUC_PrePost_All.pdf'.format(cellname))
dfs.to_csv('All_PrePost_keygenes.xls',sep='\t',index=False)

import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata=adata[adata.obs['pre_post']=='Pre',:]
adata=adata[adata.obs['tissue']=='brain',:]
adata=adata[adata.obs['Combined_outcome']!='UT',:]
lists=[]
for i in list(adata.obs['Combined_outcome']):
    if i=='Favourable':
        lists.append('R')
    else:
        lists.append('N')
adata.obs['response']=lists
adata2=adata.copy()
dfo=pd.read_table('celltype_PreResponse_brain_keygenes.xls',sep='\t')
cells=['Malignant', 'T_cell']
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
for i in cells:
    plt.figure(figsize=(8, 6))
    x=0
    for j in toplist:
        dfx=dfo[dfo['cluster']==i]
        dfx=dfx.reset_index(drop=True)
        dfx=dfx.head(j)
        genelist=list(dfx['Feature']) 
        adatas=adata2[adata2.obs['author_cell_type_update']==i,:]
        adatas=adatas[:, adatas.var_names.isin(genelist)]
        exp=adatas.X
        label=list(adatas.obs['response'])
        accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
        cellname=i.replace(' ', '_')
        dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=clist[x],lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top{}'.format(j))
        x+=1
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig('AUC_{}_PreResponse_brain_top.pdf'.format(cellname))       

dfs=pd.read_table('All_PreResponse_brain_keygenes.xls',sep='\t')
plt.figure(figsize=(8, 6))
x=0
for j in toplist:
    dfx=dfo[dfo['cluster']==i]
    dfx=dfx.reset_index(drop=True)
    dfx=dfx.head(j)
    genelist=list(dfx['Feature'])
    adatas=adata2[:, adata2.var_names.isin(genelist)]
    exp=adatas.X
    label=list(adatas.obs['response'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top{}'.format(j))
    x+=1
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('AUC_All_PreResponse_brain_top.pdf'.format(cellname))


import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata=adata[adata.obs['pre_post']=='Pre',:]
adata=adata[adata.obs['tissue']=='brain',:]
adata=adata[adata.obs['Combined_outcome']!='UT',:]
lists=[]
for i in list(adata.obs['Combined_outcome']):
    if i=='Favourable':
        lists.append('R')
    else:
        lists.append('N')
adata.obs['response']=lists
adata2=adata.copy()
dflist=[]
cells=['Malignant', 'T_cell']
for i in cells:
    adatas=adata2[adata2.obs['author_cell_type_update']==i,:]
    exp=adatas.X
    label=list(adatas.obs['response'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')+'_PostResponse_brain_'
    dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
    xtmp=dfs.head(20)
    xtmp=xtmp.loc[:,['Feature']]
    xtmp.to_csv('genelist_{}.txt'.format(i),sep='\t',index=False,header=False)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
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
    dfs['cluster']=[i]*len(dfs)
    dflist.append(dfs)
    plt.savefig('AUC_PreResponse_brain_{}.pdf'.format(cellname))
dfo=pd.concat(dflist)
dfo.to_csv('celltype_PreResponse_brain_keygenes.xls',sep='\t',index=False)

exp=adata2.X
label=list(adata2.obs['response'])
accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
cellname=i.replace(' ', '_')
dfs=Shap(model3,X_test2,adata2,y_pred_label,label,'All_PreResponse_brain')
xtmp=dfs.head(20)
xtmp=xtmp.loc[:,['Feature']]
xtmp.to_csv('genelist_all.txt',sep='\t',index=False,header=False)
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
plt.savefig('AUC_PreResponse_brain_All.pdf'.format(cellname))
dfs.to_csv('All_PreResponse_brain_keygenes.xls',sep='\t',index=False)

import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata=adata[adata.obs['pre_post']=='Pre',:]
adata=adata[adata.obs['Combined_outcome']!='UT',:]
lists=[]
for i in list(adata.obs['Combined_outcome']):
    if i=='Favourable':
        lists.append('R')
    else:
        lists.append('N')
adata.obs['response']=lists
adata2=adata.copy()
dflist=[]
cells=['Malignant', 'T_cell']
for i in cells:
    adatas=adata2[adata2.obs['author_cell_type_update']==i,:]
    exp=adatas.X
    label=list(adatas.obs['response'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')
    dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
    xtmp=dfs.head(20)
    xtmp=xtmp.loc[:,['Feature']]
    xtmp.to_csv('genelist_{}.txt'.format(i),sep='\t',index=False,header=False)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
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
    dfs['cluster']=[i]*len(dfs)
    dflist.append(dfs)
    plt.savefig('AUC_{}.pdf'.format(cellname))
dfo=pd.concat(dflist)
dfo.to_csv('celltype_keygenes.xls',sep='\t',index=False)

exp=adata2.X
label=list(adata2.obs['response'])
accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
cellname=i.replace(' ', '_')
dfs=Shap(model3,X_test2,adata2,y_pred_label,label,'All')
xtmp=dfs.head(20)
xtmp=xtmp.loc[:,['Feature']]
xtmp.to_csv('genelist_all.txt',sep='\t',index=False,header=False)
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
plt.savefig('AUC_All.pdf'.format(cellname))
dfs.to_csv('All_keygenes.xls',sep='\t',index=False)

import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

choselist=['Metastatic']
adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata=adata[adata.obs['Primary_or_met'].isin(choselist),:]
adata=adata[adata.obs['Combined_outcome']=='UT',:]
adata2=adata.copy()
dfo=pd.read_table('celltype_skinbrain_keygenes.xls',sep='\t')
cells=['Malignant', 'T_cell']
toplist=[20,10,5]
clist=['#F39900','#007900','#7B0000']
for i in cells:
    plt.figure(figsize=(8, 6))
    x=0
    for j in toplist:
        dfx=dfo[dfo['cluster']==i]
        dfx=dfx.reset_index(drop=True)
        dfx=dfx.head(j)
        genelist=list(dfx['Feature']) 
        adatas=adata2[adata2.obs['author_cell_type_update']==i,:]
        adatas=adatas[:, adatas.var_names.isin(genelist)]
        exp=adatas.X
        label=list(adatas.obs['tissue'])
        accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
        cellname=i.replace(' ', '_')
        dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=clist[x],lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top{}'.format(j))
        x+=1
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig('AUC_{}_skinbrain_top.pdf'.format(cellname))       

dfs=pd.read_table('All_skinbrain_keygenes.xls',sep='\t')
plt.figure(figsize=(8, 6))
x=0
for j in toplist:
    dfx=dfo[dfo['cluster']==i]
    dfx=dfx.reset_index(drop=True)
    dfx=dfx.head(j)
    genelist=list(dfx['Feature'])
    adatas=adata2[:, adata2.var_names.isin(genelist)]
    exp=adatas.X
    label=list(adatas.obs['tissue'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=clist[x], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'+'top{}'.format(j))
    x+=1
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('AUC_All_skinbrain_top.pdf'.format(cellname))


import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 

def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

choselist=['Metastatic']
adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata=adata[adata.obs['Primary_or_met'].isin(choselist),:]
adata=adata[adata.obs['Combined_outcome']=='UT',:]
adata2=adata.copy()
dflist=[]
cells=['Malignant', 'T_cell']
for i in cells:
    adatas=adata2[adata2.obs['author_cell_type_update']==i,:]
    exp=adatas.X
    label=list(adatas.obs['tissue'])
    accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
    cellname=i.replace(' ', '_')+'_skinbrain_'
    dfs=Shap(model3,X_test2,adatas,y_pred_label,label,cellname)
    xtmp=dfs.head(20)
    xtmp=xtmp.loc[:,['Feature']]
    xtmp.to_csv('genelist_{}.txt'.format(i),sep='\t',index=False,header=False)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve
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
    dfs['cluster']=[i]*len(dfs)
    dflist.append(dfs)
    plt.savefig('AUC_skinbrain_{}.pdf'.format(cellname))
dfo=pd.concat(dflist)
dfo.to_csv('celltype_skinbrain_keygenes.xls',sep='\t',index=False)

exp=adata2.X
label=list(adata2.obs['tissue'])
accuracy2,model3,X_test2,y_pred2,y_prob,y_pred_label,y_test=TumorClassify(exp,label)
cellname=i.replace(' ', '_')
dfs=Shap(model3,X_test2,adata2,y_pred_label,label,'All_skinbrain')
xtmp=dfs.head(20)
xtmp=xtmp.loc[:,['Feature']]
xtmp.to_csv('genelist_all.txt',sep='\t',index=False,header=False)
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
plt.savefig('AUC_skinbrain_All.pdf'.format(cellname))
dfs.to_csv('All_skinbrain_keygenes.xls',sep='\t',index=False)

import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib as mpl 
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb 
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc 
from adjustText import adjust_text

def vocalno(data,fold_change_threshold, pvalue_threshold,vals,text,gene_col,xlim,ylim):
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_axes([0.13,0.1,0.5,0.7])
    fold_change_threshold = fold_change_threshold  # Absolute log2 fold change > 1
    pvalue_threshold = pvalue_threshold    # p-value < 0.05
    data["Significance"] = "Not Significant"
    data.loc[data[vals]<pvalue_threshold,"Significance"]="Significant"
    data.loc[(data["log2foldchange"] > fold_change_threshold) & (data[vals] < pvalue_threshold), "Significance"] = "Significant up"
    data.loc[(data["log2foldchange"] < -fold_change_threshold) & (data[vals] < pvalue_threshold), "Significance"] = "Significant down"
    data["-log10(pvalue)"] = -np.log10(data[vals])
    data["-log10(pvalue)"] = data["-log10(pvalue)"].clip(0,ylim)
    colors = {"Significant up": "#EC0000", "Not Significant": "#DBDBDB","Significant down":"#0080F6","Significant":"#FCECCC"}
    data=data.reset_index(drop=True)
    draw_colors=[]
    for i in range(len(data)):
        draw_colors.append(colors[data['Significance'][i]])
    plt.scatter(data["log2foldchange"], data["-log10(pvalue)"], c=draw_colors, alpha=0.5,edgecolors='none')
    # Add horizontal and vertical lines for thresholds
    plt.axhline(-np.log10(pvalue_threshold), color="blue", linestyle="--", linewidth=1, label="p-value threshold")
    plt.axvline(-fold_change_threshold, color="green", linestyle="--", linewidth=1, label="log2FC threshold")
    plt.axvline(fold_change_threshold, color="green", linestyle="--", linewidth=1)
    plt.xlim(-xlim,xlim)
    plt.ylim(0,ylim)
    for label, color in colors.items():
        plt.scatter([], [], color=color, label=label)
    # Titles and labels
    plt.title("Volcano Plot")
    plt.xlabel("log2(Fold Change)")
    plt.ylabel("-log10(p-value)")
    plt.legend(frameon=False, loc="upper right",bbox_to_anchor=(1.7, 1))
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    text=pd.DataFrame({gene_col:text})
    label_data=pd.merge(data,text,how='inner',on=[gene_col])
    plt.grid(alpha=0)
    plt.tight_layout()
    texts = []
    for _, row in label_data.iterrows():
        texts.append(plt.text(
        row['log2foldchange'],
        row['-log10(pvalue)'],
        row[gene_col],
        fontsize=8,
        color='black',
        ha='center',
        va='center'))
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle='-', color='black', lw=0.5),  # 添加连接线
        expand_text=(1.2, 1.2),  # 扩展文本间距
        expand_points=(1.2, 1.2),  # 扩展点周围区域
        force_text=0.5,  # 调整力度
        force_points=0.5,
        lim=100)
    return plt


def SamplePick(adata):
    tmp1=adata.obs[adata.obs['response']=='R']
    tmp2=adata.obs[adata.obs['response']=='N']
    ratio=len(tmp1)/len(adata.obs)
    lists=[]
    if len(tmp1)>100 and len(tmp2)>100 and ratio>0.25:
        return adata
    else:
        return None

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

def QcNorm(adata):
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize counts
    sc.pp.log1p(adata)  # Log-transform the data
    sc.pp.scale(adata, max_value=10)
    return adata

adata=sc.read_h5ad('Melanoma_ICB_scRNA.h5ad')
adata=adata[adata.obs['pre_post']=='Pre',:]
adata=adata[adata.obs['Combined_outcome']!='UT',:]
lists=[]
for i in list(adata.obs['Combined_outcome']):
    if i=='Favourable':
        lists.append('R')
    else:
        lists.append('N')
adata.obs['response']=lists


group1='R'
group2='N'
cells=['Malignant', 'T_cell']
df=pd.read_table('celltype_keygenes.xls',sep='\t')
genelist=[]
for i in cells:
    dfs=df[df['cluster']==i]
    dfs=dfs.head(20)
    lists=list(dfs['Feature'])
    genelist.append(lists)
dfv=adata.var
dfv['genename']=list(dfv.index)
dfv=dfv.reset_index(drop=True)
dfv=dfv.loc[:,['ensembl','genename']]
vals='q_value'
xlims=[100,100]
ylims=[100,10]
x=0
for i in cells: 
    adatas=adata[adata.obs['author_cell_type_update']==i,:]
    sc.tl.rank_genes_groups(adatas,groupby="response", groups=[group1],reference=group2,method="wilcoxon",key_added="rank_genes_groups")
    results = adatas.uns["rank_genes_groups"]
    gene_names = results["names"][group1]  # Gene names
    log2fc = results["logfoldchanges"][group1]  # Log2 fold changes
    pvals_adj = results["pvals_adj"][group1]  # Adjusted p-values (q-values)
    differential_expression = pd.DataFrame({"gene": gene_names,"log2foldchange": log2fc,"q_value": pvals_adj})
    differential_expression = differential_expression.sort_values(by="q_value")
    differential_expression.to_csv("Log2FC_{}.csv".format(i), index=False)
    differential_expression.columns=['ensembl','log2foldchange','q_value']
    differential_expression=pd.merge(differential_expression,dfv)
    text=genelist[x]
    xlim=xlims[x]
    ylim=ylims[x]
    differential_expression['log2foldchange']=differential_expression['log2foldchange'].clip(-xlim, xlim)
    plt=vocalno(differential_expression,2, 0.01,vals,text,'genename',xlim,ylim)
    plt.savefig('{}_singlecell_volcanotag.pdf'.format(i))
    x+=1


xlim=20
ylim=30
adatas=adata
sc.tl.rank_genes_groups(adatas,groupby="response", groups=[group1],reference=group2,method="wilcoxon",key_added="rank_genes_groups")
results = adatas.uns["rank_genes_groups"]
gene_names = results["names"][group1]  # Gene names
log2fc = results["logfoldchanges"][group1]  # Log2 fold changes
pvals_adj = results["pvals_adj"][group1]  # Adjusted p-values (q-values)
differential_expression = pd.DataFrame({"gene": gene_names,"log2foldchange": log2fc,"q_value": pvals_adj})
differential_expression = differential_expression.sort_values(by="q_value")
differential_expression.to_csv("Log2FC_{}.csv".format('all'), index=False)
differential_expression.columns=['ensembl','log2foldchange','q_value']
differential_expression=pd.merge(differential_expression,dfv)
df=pd.read_table('All_keygenes.xls',sep='\t')
df=df.head(20)
text=list(df['Feature'])
differential_expression['log2foldchange']=differential_expression['log2foldchange'].clip(-xlim, xlim)
plt=vocalno(differential_expression,2, 0.01,vals,text,'genename',xlim,ylim)
plt.savefig('{}_singlecell_volcanotag.pdf'.format('all'))





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
#import scrublet as scr
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

def zscore(adata):
    sc.pp.scale(adata, zero_center=True, max_value=10)    
    return adata

def Train_Pre(adata,cancer1,cancer2,gene_index,label):
    adata_train = adata[adata.obs['disease'].isin(cancer1)].copy()
    adata_test = adata[adata.obs['disease'].isin(cancer2)].copy()
    adata_train = adata_train[:, adata_train.var.index.isin(gene_index)].copy()
    adata_test = adata_test[:, adata_test.var.index.isin(gene_index)].copy()    
    adata_train=zscore(adata_train)
    adata_test=zscore(adata_test)
    X_train=adata_train.X
    X_test=adata_test.X
    y_train=adata_train.obs[label]
    y_test=adata_test.obs[label]
    return X_train,X_test,y_train,y_test

def TumorClassify(X_train,X_test,y_train,y_test):
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    y_pred_label = y_pred
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test

def Shap(model,X_test,adata,y_pred,name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({'Feature': list(adata.var['GeneName']),'Mean Absolute SHAP Value': mean_abs_shap})
    feature_importance_df = feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=adata.var['GeneName'],plot_type="bar" ,show=False)
    plt.title("Summary Plot")
    plt.savefig('shap_summary_{}.pdf'.format(name),bbox_inches="tight")  # Save to PDF
    plt.close()
    return feature_importance_df

def draw_AUC(y_test,y_prob,name):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='#F39900', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig('AUC_{}_top.pdf'.format(name))  

adata=sc.read_h5ad('ICB.h5ad')
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:, adata.var.highly_variable].copy()
gene_index=list(adata.var.index)

adata=sc.read_h5ad('ICB.h5ad')
adata=adata[adata.obs['pre_post']=='Pre']
adata=adata[adata.obs['Combined_outcome'].isin(['Favourable','Unfavourable'])].copy()
lists=[]
for i in list(adata.obs['Combined_outcome']):
    if i=='Unfavourable':
        lists.append(0)
    else:
        lists.append(1)
adata.obs['treatment_outcome']=lists


cancer1=['triple-negative breast carcinoma','estrogen-receptor positive breast cancer','HER2 positive breast carcinoma','intrahepatic cholangiocarcinoma']
cancer2=['metastatic melanoma','melanoma']
label='treatment_outcome'

X_train,X_test,y_train,y_test=Train_Pre(adata,cancer1,cancer2,gene_index,label)
accuracy,model,X_test,y_pred,y_prob,y_pred_label,y_test=TumorClassify(X_train,X_test,y_train,y_test)
name='ICB_Cross1'
lists=[]
for i in list(adata.var['feature_name']):
    if '_' in i:
        i=i.split('_')[0]
        lists.append(i)
    else:
        lists.append(i)
adata.var['GeneName']=lists
adata= adata[:, adata.var.index.isin(gene_index)].copy()
dfa=Shap(model,X_test,adata,y_pred,name)
draw_AUC(y_test,y_prob,name)
plt.close()

adata=sc.read_h5ad('ICB.h5ad')
adata=adata[adata.obs['pre_post']=='Pre']
adata=adata[adata.obs['Combined_outcome'].isin(['Favourable','Unfavourable'])].copy()
lists=[]
for i in list(adata.obs['Combined_outcome']):
    if i=='Unfavourable':
        lists.append(0)
    else:
        lists.append(1)
adata.obs['treatment_outcome']=lists
adata=adata[adata.obs['cell_type'].str.contains('T')==True]
adata.layers['log']=adata.X.copy()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.scale(adata, max_value=10)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:, adata.var.highly_variable].copy()
sc.tl.pca(adata, n_comps=30)
sc.external.pp.harmony_integrate(adata, key='batch', max_iter_harmony=10)
sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_neighbors=15)
sc.tl.leiden(adata, resolution=0.4)
sc.tl.umap(adata)
fig = plt.figure(figsize=(14,10))
reduced_data = adata.obsm['X_pca']
metadata = adata.obs[['disease','Sample']].astype(str).copy()
fig = plt.figure(figsize=(15,10))
harmony_out = hm.run_harmony(
        data_mat = reduced_data,
        meta_data = metadata,
        vars_use = ['disease','Sample'],
        max_iter_harmony = 20,
        epsilon_harmony = 1e-4)
adata.obsm['X_pca_harmony'] = harmony_out.Z_corr.T
sc.pp.neighbors(adata, use_rep='X_pca_harmony')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, key_added='leiden_har')
fig = plt.figure(figsize=(9,4))
ax1 = fig.add_axes([0.05,0.1,0.25,0.75])
ax2 = fig.add_axes([0.65,0.1,0.25,0.75])
sc.pl.umap(adata, color='disease', ax=ax1,
               show=False, title="After Batch Effect Removal")
sc.pl.umap(adata, color='leiden_har', ax=ax2,
               show=False, title="After Batch Effect Removal")
axlist=[ax1,ax2]
for ax in axlist:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,labels,ncol=1,loc='center left',bbox_to_anchor=(1.01, 0.5),fontsize='small', frameon=False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
plt.savefig('PanCancerTcells.pdf')
adata.write('PanCancerTcells.h5ad')

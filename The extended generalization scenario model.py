import pandas as pd
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc,  precision_score, recall_score, f1_score 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from joblib import dump, load

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from tqdm import tqdm

file="./CytoplasmUniqueGeneSymbol.statistic.csv"
csvPD = pd.read_csv(file)
csv = csvPD
train_csv = csv

file1 = "./ALL.KEGG.UP.csv"
file2 = "./ALL.REACTOME.UP.csv"
file3 = "./ALL.UP.BP.csv"
file4 = "./ALL.UP.CC.csv"
file5 = "./ALL.UP.MF.csv"

xls1 = pd.read_csv(file1,sep='\t')
xls2 = pd.read_csv(file2,sep='\t')
xls3 = pd.read_csv(file3,sep='\t')
xls4 = pd.read_csv(file4,sep='\t')
xls5 = pd.read_csv(file5,sep='\t')

xls_all = pd.concat([xls3,xls4,xls5],axis=0)
xls_all = xls_all.reset_index(drop=True)
xls_all['geneID'] = xls_all['geneID'].replace("[", "").replace("]", "").replace("'",'').replace(" ","").replace("/",",")


flag = 0

for i in xls_all['geneID']:
    # print(i)
    xls_all['geneID'][flag] = xls_all['geneID'][flag].replace("[", "").replace("]", "").replace("'",'').replace(" ","").replace("/",",")
    xls_all['geneID'][flag] = xls_all['geneID'][flag].split(',')
    flag = flag + 1
    if(flag == 2960):
        break
    # print(xls_all['geneID'][flag])


sign = set()
# with tqdm(total=len(csv)) as pbar:
for i in tqdm(csv['symbol']):
    for j in xls_all['geneID']:
        if i in j:
            sign.add(i)

xls_all['label'] = xls_all.apply(lambda row:0, axis=1)


new_csv = csv
new_csv['pathway'] = new_csv.apply(lambda row: [], axis=1)
pathway = {}
flag = 0
for i,j in tqdm(new_csv.iterrows()):
    # print(i)
    # print(j['symbol'])
    flag = flag + 1
    protein_name = str(j['x'])
    symbol = str(j['symbol'])
    pathway[protein_name] = {}
    pathway[protein_name][symbol] = []
    for k in xls_all.iterrows():
        if j['symbol'] in k[1]['geneID']:
            # print(j['symbol'])
            # print(k[1]['Description'])
            # pathway[protein_name][].append(k[1]['Description'])
            pathway[protein_name][symbol].append(k[1]['Description'])
            j['pathway'].append(k[1]['Description'])
    if flag % 1000 == 0:
        print(flag)
    if flag == 58767:
        break
new_csv.to_csv("./pathway.csv", index=False)


file="./pathway.csv"
csvPD=pd.read_csv(file)
csv = csvPD
new_csv = csv
# pathway_csv.head(30)

flag = 0
for i in new_csv['pathway']:
    if len(i) != 2:
        flag = flag + 1
print(flag)

one_set = set()
one_set.add('UCKL1')
flag = 0
for i in xls_all['geneID']:
    tem_set = set()
    if len(i) < 11 :
        for j in one_set:
            if j in i:
                for k in i:
                    tem_set.add(k)
    one_set = one_set.union(tem_set)


is_not_empty = new_csv['pathway'].apply(lambda x: len(x) > 2)

# 使用布尔型Series筛选出满足条件的行
new_df = new_csv[is_not_empty]
new_df = new_df.reset_index(drop=True)
new_df['label'] = new_df.apply(lambda row:0, axis=1)
new_df['MV'] = (new_df['MV'] - new_df['MV'].mean()) / new_df['MV'].std()
new_df['zi'] = (new_df['zi'] - new_df['zi'].mean()) / new_df['zi'].std()
new_df['pHscore'] = (new_df['pHscore'] - new_df['pHscore'].mean()) / new_df['pHscore'].std()
new_df['NetCharge'] = (new_df['NetCharge'] - new_df['NetCharge'].mean()) / new_df['NetCharge'].std()
new_df['Hydropathy'] = (new_df['Hydropathy'] - new_df['Hydropathy'].mean()) / new_df['Hydropathy'].std()
new_df.drop('pathway', axis=1, inplace=True)

flag = 0
for i in new_df['symbol']:
    if i in one_set:
        new_df['label'][flag] = 1
    flag = flag + 1
flag = 0
for i in new_df['label']:
    if i == 1:
        flag = flag + 1

count_classes = new_df.value_counts(new_df['label'],sort=True) # 目标变量正负样本的分布
print(count_classes)

add = new_df
add['length'] = (add['length'] - add['length'].mean()) / add['length'].std()
# drop_list = ['x','ensmebl_p','ensmebl_t','ensmebl_g','symbol','Q','S','T','U','R','H','K']
# drop_list = ['A','V','L','I','M','P','F','W']
drop_list = ['x','ensmebl_p','ensmebl_t','ensmebl_g','symbol','Q','S','T','U','R','H','K','D','E','C','G','N','Y','A','V','L','I','M','P','F','W']
add.drop(drop_list, axis=1, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(new_df, new_df['label'], test_size=0.4, random_state=0)
X_train.drop('label', axis=1, inplace=True)
X_test.drop('label', axis=1, inplace=True)
print('过采样前，1的样本的个数为：',len(y_train[y_train==1]))
print('过采样前，0的样本的个数为：',len(y_train[y_train==0]))
over_sampler=SMOTE(random_state=0)
X_os_train,y_os_train=over_sampler.fit_resample(X_train,y_train)
print('过采样后，1的样本的个数为：',len(y_os_train[y_os_train==1]))
print('过采样后，0的样本的个数为：',len(y_os_train[y_os_train==0]))
column_trans = Pipeline([('scaler', StandardScaler())])
preprocessing = ColumnTransformer(
    transformers=[
        ('column_trans', column_trans, ['Hydropathy','MV','zi','pHscore','NetCharge','length'])
    ], remainder='passthrough')
over_pipe_add = Pipeline([
    ('preprocessing', preprocessing),
    ('sampler', SMOTE() ),
    ('classifier', RandomForestClassifier(n_estimators=10))
])
over_pipe_add.fit(X_train, y_train)

y_pred = over_pipe_add.predict(X_test)
p = precision_score(y_test, y_pred)
r = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("precision（准确率）: ", p)



import joblib

joblib.dump(over_pipe_add, './new_over_pipe.joblib')  # 模型保存

# 获取随机森林分类器
# 特征重要性
rf_classifier = over_pipe_add.named_steps['classifier']
feature_importances = rf_classifier.feature_importances_


feature_names = ['Hydropathy', 'MV', 'zi', 'pHscore', 'NetCharge', 'length'] + list(new_df.columns.drop(['label']).difference(['Hydropathy','MV','zi','pHscore','NetCharge','length']))

# 直接排序后显示
sorted_features = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_features:
    print(f" {feature}:{importance:.3f}")

#绘制混淆矩阵
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # 使用seaborn进行美化
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

#模型预测保存结果
file="./CytoplasmUniqueGeneSymbol.statistic.csv"
csvPD = pd.read_csv(file)
csv = csvPD
test_csv = csv
# new_csv.drop(['label'],axis = 1 ,inplace = True)
result_csv = new_csv
columns_to_keep = ['length','Hydropathy','MV','zi','pHscore','NetCharge']
test_csv['length'] = (test_csv['length'] - test_csv['length'].mean()) / test_csv['length'].std()
# 在原DataFrame上操作，删除其他列
test_csv  = test_csv [columns_to_keep]
test_csv['MV'] = (test_csv['MV'] - test_csv['MV'].mean()) / test_csv['MV'].std()
test_csv['zi'] = (test_csv['zi'] - test_csv['zi'].mean()) / test_csv['zi'].std()
test_csv['pHscore'] = (test_csv['pHscore'] - test_csv['pHscore'].mean()) / test_csv['pHscore'].std()
test_csv['NetCharge'] = (test_csv['NetCharge'] - test_csv['NetCharge'].mean()) / test_csv['NetCharge'].std()
test_csv['Hydropathy'] = (test_csv['Hydropathy'] - test_csv['Hydropathy'].mean()) / test_csv['Hydropathy'].std()
test_csv['length'] = (test_csv['length'] - test_csv['length'].mean()) / test_csv['length'].std()
test_csv['label'] = 0

pipeline01 = joblib.load('./new_over_pipe.joblib')
result01 = pipeline01.predict(test_csv)
result = []
test_csv['label'] = result
test_csv.to_csv("./output.csv", index=False)
test_csv
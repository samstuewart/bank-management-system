#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
df = pd.read_excel("bank-full.xlsx")
df.head()
df.shape
df.isnull().sum()
df.describe()
df.dtypes
df.info()
sns.boxplot(y=df['age'],
 x=df['education'])
sns.histplot(df['job'],bins=10)
sns.violinplot(y=df['age'],
 x=df['education'])
sns.violinplot(y=df['age'],
 x=df['default'])
correlation=df.corr()
plt.figure(figsize=(7,7))
sns.heatmap(correlation,cbar=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap="PuB
uGn_r")
sns.pairplot(df,hue='default')
df.columns
#unique values
print(df['job'].unique())
print(df['marital'].unique())
print(df['education'].unique())
print(df['default'].unique())
print(df['housing'].unique())
print(df['loan'].unique())
print(df['contact'].unique())
print(df['month'].unique())
print(df['day'].unique())
 27
LABEL ENCODING
 
from sklearn.preprocessing import LabelEncoder #to convert a categorical data to 
numerical data
le_age = LabelEncoder()
le_job = LabelEncoder()
le_marital = LabelEncoder()
le_education = LabelEncoder()
le_default = LabelEncoder()
le_balance = LabelEncoder()
le_housing = LabelEncoder()
le_loan = LabelEncoder()
le_contact = LabelEncoder()
le_day = LabelEncoder()
le_month = LabelEncoder()
le_duration = LabelEncoder()
le_campaign = LabelEncoder()
le_pdays = LabelEncoder()
le_previous = LabelEncoder()
le_poutcome = LabelEncoder()
df['age'] = le_age.fit_transform(df['age'])
df['job'] = le_job.fit_transform(df['job'])
df['marital'] = le_marital.fit_transform(df['marital'])
df['education'] = le_education.fit_transform(df['education'])
df['default'] = le_default.fit_transform(df['default'])
df['balance'] = le_balance.fit_transform(df['balance'])
df['housing'] = le_housing.fit_transform(df['housing'])
df['loan'] = le_loan.fit_transform(df['loan'])
df['contact'] = le_contact.fit_transform(df['contact'])
df['day'] = le_day.fit_transform(df['day'])
df['month'] = le_month.fit_transform(df['month'])
df['duration'] = le_duration.fit_transform(df['duration'])
df['campaign'] = le_campaign.fit_transform(df['campaign'])
df['pdays'] = le_pdays.fit_transform(df['pdays'])
df['previous'] = le_previous.fit_transform(df['previous'])
df['poutcome'] = le_poutcome.fit_transform(df['poutcome'])
df.head()
df.columns
 28
X = df.loc[:,['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
 'previous', 'poutcome']]
y = df.loc[:,'y']
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,train_size=0.8,test_size=0.2)
print(Xtrain.shape)
print(Xtest.shape)
print(ytrain.shape)
print(ytest.shape)
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(Xtrain,ytrain)
ypred = model.predict(Xtest)
ypred
Xpred = model.predict(Xtrain)
Xpred
Ytest
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
print(sklearn.metrics.classification_report(ytest,ypred))
print(sklearn.metrics.confusion_matrix(ytest,ypred))
client=pd.DataFrame(ypred) #no-number of people not subscribed ,yes- number of people
subscribed
client.value_counts()
print(sklearn.metrics.accuracy_score(ytest,ypred))

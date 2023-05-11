# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 12:20:08 2023

@author: roshn
https://www.kaggle.com/code/gauravduttakiit/titanic-survivors-logistic-regression-model#Step-1:-Importing-and-Merging-Data
"""

#  Step 1: Importing and Merging Data
import warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, seaborn as sns,matplotlib.pyplot as plt

train= pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()

train.shape
train.info()

train.describe()



test=pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()

test.shape
test.info()
test.describe()

#  Correcting Datatype for the variable in data

test.info()
train.info()

train['Pclass'].value_counts(ascending=False)
#PClass 

train['Pclass'].value_counts(ascending=False)
# it is a categorical data

print("Train['Pclass']  : ",train.Pclass.dtype)
print("Test['Pclass']   : ",test.Pclass.dtype)

train['Pclass']=train['Pclass'].astype('object')
test['Pclass']=test['Pclass'].astype('object')

print("Train['Pclass']  : ",train.Pclass.dtype)
print("Test['Pclass']   : ",test.Pclass.dtype)

#Survived
train['Survived'].value_counts(ascending=False)
# it is a categorical data

#Variable are correctly classified & added in dataset


#Decoding Values
# Pclass

train['Pclass'] = train['Pclass'].replace({ 1 : '1st', 2: '2nd',3: '3rd'}).astype('category')
test['Pclass'] = test['Pclass'].replace({ 1 : '1st', 2: '2nd',3: '3rd'}).astype('category')
 
 
train['Pclass'].value_counts(ascending=False) 
test['Pclass'].value_counts(ascending=False)

# Embarked
train['Embarked'] = train['Embarked'].replace({ 'C' : 'Cherbourg', 'Q': 'Queenstown','S': 'Southampton'}).astype('object')
test['Embarked'] = test['Embarked'].replace({ 'C' : 'Cherbourg', 'Q': 'Queenstown','S': 'Southampton'}).astype('object')

train['Embarked'].value_counts(ascending=False)

test['Embarked'].value_counts(ascending=False)

#Variable are all decoded & added in dataset

#Step 2: Inspecting the Dataframe
train.head()
train.shape
train.describe()
train.info()

#We found few missing values in few of Columns
round(test.isnull().sum()*100/len(test),2)
test.isnull().sum()

round(train.isnull().sum()*100/len(train),2)
train.isnull().sum()

#Input data for missing data
# Cabin
round(test.Cabin.isnull().sum()*100/len(test),2)
test.Cabin.isnull().sum()

pd.set_option('display.max_rows', None)
test.Cabin.value_counts(ascending=False)

plt.figure(figsize=(30,8))
sns.countplot(x='Cabin',data=test)
plt.xticks(rotation=90)
plt.show()

# Let replace missing value with a variable X, which means it's Unknown
test['Cabin'] = test['Cabin'].replace(np.nan,'X')
test['Cabin'].isnull().sum()
train.Cabin.value_counts(ascending=False)


plt.figure(figsize=(32,8))
ax=sns.countplot(x='Cabin',data=test)
ax.set_yscale('log')
plt.xticks(rotation=90)
plt.show()

#Similarly, For train data
round(train.Cabin.isnull().sum()*100/len(train),2)
train.Cabin.isnull().sum()
pd.set_option('display.max_rows', None)
train.Cabin.value_counts(ascending=False)

plt.figure(figsize=(30,8))
sns.countplot(x='Cabin',data=train)
plt.xticks(rotation=90)
plt.show()

#Let replace missing value with a variable X, which means it's Unknown
train['Cabin'] = train['Cabin'].replace(np.nan,'X')
train['Cabin'].isnull().sum()
test.Cabin.value_counts(ascending=False)

plt.figure(figsize=(32,8))
ax=sns.countplot(x='Cabin',data=train)
ax.set_yscale('log')
plt.xticks(rotation=90)
plt.show()

#Fare
round(test.Fare.isnull().sum()*100/len(test),2)
test.Fare.isnull().sum()
test.Fare.describe()


plt.figure(figsize=(8,8))
sns.violinplot(x='Fare',data=test)
plt.show()    

#There are outliers for this variable, hence, Median is prefered over mean

test['Fare'] = test['Fare'].replace(np.nan,train.Fare.median())
test.Fare.isnull().sum()

#Age
round(train.Age.isnull().sum()*100/len(train),2)
train.Age.isnull().sum()
train.Age.describe()

plt.figure(figsize=(8,8))
sns.violinplot(x='Age',data=train)
plt.show()

#There are outliers for this variable, hence, Median is prefered over mean
train['Age'] = train['Age'].replace(np.nan,train.Age.median())
train['Age'].isnull().sum()

round(test.Age.isnull().sum()*100/len(test),2)
test.Age.isnull().sum()
test.describe()

plt.figure(figsize=(8,8))
sns.violinplot(x='Age',data=test)
plt.show()

#There are outliers for this variable, hence, Median is prefered over mean
test['Age'] = test['Age'].replace(np.nan,train.Age.median())
test['Age'].isnull().sum()

#Emarked
round(train.Embarked.isnull().sum()*100/len(train),2)
train.Embarked.isnull().sum()

train.Embarked.value_counts(ascending=False)
#Since, it's catergorical datatype, we opt for Mod for null values
train['Embarked'] = train['Embarked'].replace(np.nan,train.Embarked.mode()[0])

train.Embarked.mode()
train.Embarked.isnull().sum()

#Final Check


train.isnull().sum()
test.isnull().sum()

# No Nan records are avilable in data sets

#EDA
#Checking Correlation Matrix

#Heatmap
plt.figure(figsize = (10,10))
sns.heatmap(train.corr(),annot = True,cmap="tab20c")
plt.show()

#Pairplot
sns.pairplot(train)
plt.show()

#Count plot
plt.figure(figsize=(8,8))
ax = sns.countplot(x='Pclass',data=train,hue="Survived")
bars = ax.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    ax.text(left.get_x() + left.get_width()/2., height_l + 10, '{0:.0%}'.format(height_l/total), ha="center")
    ax.text(right.get_x() + right.get_width()/2., height_r + 10, '{0:.0%}'.format(height_r/total), ha="center")


#insights 
#.....
#........


# Premium cost increased the chance of survival in that acciden

plt.figure(figsize=(8,8))
ax = sns.countplot(x='Sex',data=train,hue="Survived")
bars = ax.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    ax.text(left.get_x() + left.get_width()/2., height_l + 10, '{0:.0%}'.format(height_l/total), ha="center")
    ax.text(right.get_x() + right.get_width()/2., height_r + 10, '{0:.0%}'.format(height_r/total), ha="center")


# Insights
#............
#
#...........
# gender increased the chance for survival in that accident

#Age

plt.figure(figsize=(8,8))
sns.violinplot(y='Age',x='Survived',hue='Survived',data=train)
plt.show()
#Mean Age for people who survived is 28 years, which is less compared with Mean Age for people who didn't survived is 30 years.
#Median , 75th percentitle is same for both cases
#We can create a column 'Family' which will store values of sibsp + parch,
# sibsp -> # of siblings / spouses aboard the Titanic
#parch -> # of parents / children aboard the Titanic
#& later drop these 2 columns from both dataset for uniformity


#Add new feature
train['Family']= train['SibSp']+ train['Parch']+ 1 #including the passenger him/herself
train=train.drop(['SibSp','Parch'],axis=1)
train.head()

plt.figure(figsize=(20,8))
sns.violinplot(y='Age',x='Family',hue='Survived',data=train)
# why does age selected on y -axis,  insted count?
plt.show()

train.Family[train.Survived==1].describe()

print('Percentage of People Survived with their family member count')
train.Family[train.Survived==1].value_counts()* 100/len(train)


print('Number of People Survived with their family member count')
train.Family[train.Survived==1].value_counts()

print('Perceptage in total Survival with family count as ')
train.Family[train.Survived==1].value_counts()* 100/len(train.Family[train.Survived==1])


train.Family[train.Survived==0].describe()

print('Perecentage of People Not Survived with their family member count')
train.Family[train.Survived==0].value_counts()* 100/len(train)

print('Perceptage in total Death with family count as ')
train.Family[train.Survived==0].value_counts()* 100/len(train.Family[train.Survived==0])

print('Number of People Not Survived with their family member count')
train.Family[train.Survived==0].value_counts()



plt.figure(figsize=(20,8))
ax = sns.countplot(x='Family',data=train,hue="Survived")
bars = ax.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    ax.text(left.get_x() + left.get_width()/2., height_l + 10, '{0:.0%}'.format(height_l/total), ha="center")
    ax.text(right.get_x() + right.get_width()/2., height_r + 10, '{0:.0%}'.format(height_r/total), ha="center")

#Insights

#' Add feature to Test data set'
test['Family']= test['SibSp']+ test['Parch']+ 1 #including the passenger him/herself
test=test.drop(['SibSp','Parch'],axis=1)
test.head()

#Name & Ticket Number are not an important feature for prediction
train=train.drop(['Name','Ticket'],axis=1)
train.head()

#Similarly, For testdata, we perform same action

test=test.drop(['Name','Ticket'],axis=1)
test.head()


#fare

plt.figure(figsize=(8,8))
sns.violinplot(y='Fare',x='Survived',hue='Survived',data=train)
plt.show()

train.Fare[train.Survived==1].describe()
train.Fare[train.Survived==0].describe()

#Insights

#cabin
train.Cabin.value_counts()

plt.figure(figsize=(32,8))
ax=sns.barplot(x='Cabin',y='Fare',hue='Survived',data=train)
plt.xticks(rotation=90)
plt.show()

train.Cabin[train.Survived==0].value_counts(ascending=False)*100/len(train.Cabin[train.Survived==0])


train.Cabin[train.Survived==1].value_counts(ascending=False)*100/len(train.Cabin[train.Survived==1])
#Insights


#Embarked
plt.figure(figsize=(8,8))
ax = sns.countplot(x='Embarked',data=train,hue="Survived")
bars = ax.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    ax.text(left.get_x() + left.get_width()/2., height_l + 10, '{0:.0%}'.format(height_l/total), ha="center")
    ax.text(right.get_x() + right.get_width()/2., height_r + 10, '{0:.0%}'.format(height_r/total), ha="center")



train.Embarked.value_counts()
train.Embarked.value_counts()*100/len(train)

train.Embarked[train.Survived==0].value_counts(ascending=False)

train.Embarked[train.Survived==0].value_counts(ascending=False)*100/len(train.Embarked[train.Survived==0])

train.Embarked[train.Survived==1].value_counts(ascending=False)

train.Embarked[train.Survived==1].value_counts(ascending=False)*100/len(train.Embarked[train.Survived==1])

#Insights

len(train.Cabin.unique())
# there are 148 unique values for cabin., hence not an imp field to be considered.
# So drop it from dataset.

test=test.drop(['Cabin'],axis=1)
test.head()

train=train.drop(['Cabin'],axis=1)
train.head()

##Step 3: Feature Scaling


train.info()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train[['Age', 'Fare','Family']]= scaler.fit_transform(train[['Age', 'Fare','Family']])
train.head()

test[['Age', 'Fare','Family']]= scaler.transform(test[['Age', 'Fare','Family']])
test.head()

# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(train[['Pclass', 'Sex','Embarked']], drop_first=True)

# Adding the results to the master dataframe
train = pd.concat([train, dummy1], axis=1)
train.head()

#Drop esisting colms.
train=train.drop(['Pclass', 'Sex','Embarked'],axis=1)
train.head()

# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy2 = pd.get_dummies(test[['Pclass', 'Sex','Embarked']], drop_first=True)

# Adding the results to the master dataframe
test = pd.concat([test, dummy2], axis=1)
test.head()

### Checking the Survived Rate
Survived = (sum(train['Survived'])/len(train['Survived'].index))*100
Survived

### Step 4: Looking at Correlations
train['Survived']=train['Survived'].astype('uint8')

# Let's see the correlation matrix 
plt.figure(figsize = (10,10))   
sns.heatmap(train.corr(),annot = True,cmap="tab20c")
plt.show()

####### Step 5: Model Building
y_train=train.pop('Survived')
X_train=train

X_train.head()
y_train.head()

import statsmodels.api as sm
# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()

# step 5 Feature selection using RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, 5)             
rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
col


X_train.columns[~rfe.support_]


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()

# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]

y_train_pred_final = pd.DataFrame({'Survived':y_train.values, 'Survived_Prob':y_train_pred})
y_train_pred_final['PassengerId'] = y_train.index
y_train_pred_final.head()

#Creating new column 'predicted' with 1 if Survived_Prob > 0.5 else 0

y_train_pred_final['predicted'] = y_train_pred_final.Survived_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


from sklearn import metrics
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.predicted )
print(confusion)

# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted))


#Checking VIFs
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


#Metrics beyond simply accuracy
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives

# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)

# Let us calculate specificity
TN / float(TN+FP)


#Calculate false postive rate - 
print(FP/ float(TN+FP))
# positive predictive value 
print (TP / float(TP+FP))

# Negative predictive value
print (TN / float(TN+ FN))

# rest of the steps to be reviwed.
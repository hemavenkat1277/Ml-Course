#!/usr/bin/env python
# coding: utf-8

# In[24]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ## Load the datase

# In[35]:


df = pd.read_csv("Titanic.csv")
print(df.isnull())
df.isnull().sum()


# ## Statistical Analysis

# In[20]:


print("Statistical Summary of Numeric Features:\n",df.describe())
categorical_summary={}
for col in df.select_dtypes(include=['object']).columns:
    categorical_summary[col]=df[col].value_counts()
print("\nCategorical Features - Value Counts:")
for col,count in categorical_summary.items():
    print(f"{col}:\n{count}\n")


# ## : Check the missing values and replace them with 

# In[17]:


mode_value = df['Cabin'].mode()[0] 
df['Cabin'].fillna(mode_value, inplace=True) 
df.isnull().sum()

df['Age'].fillna(df['Age'].mean(), inplace=True)
 
mode_value = df['Embarked'].mode()[0] 
df['Embarked'].fillna(mode_value, inplace=True) 

df.head()


# ## Check the outliers and drop them out

# In[39]:


plt.figure(figsize=(10,5))
sns.boxplot(x=df['Fare'])
plt.title('Fare')
plt.xlabel('Fare')
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x=df['Age'])
plt.title('Boxplot of Age')
plt.xlabel('Age')
plt.show()


# ## Find the correlation

# In[40]:


for col in df.select_dtypes(include=['int64','float64']).columns:
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3-Q1
    lower = Q1-1.5*IQR
    upper = Q3+1.5*IQR
    d=df[(df[col]>=lower) & (df[col]<=upper)]
print(df.head())


# ## Separate independent features and Target

# In[30]:


plt.figure(figsize=(10,6))
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare')
plt.xlabel('Fare')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x=df['Age'])
plt.title('Boxplot of Age')
plt.xlabel('Age')
plt.show()


# 
# 

# In[46]:


cor1 = df['Parch'].corr(df['Age'])
cor2 = df['Pclass'].corr(df['Fare'])
print(f"Correlation between 'Parch' and 'Age' : {cor1}\nBetween 'Pclass' and 'Fare' : {cor2}")


# In[48]:


x = df.drop(columns='Survived')
y = df['Survived']

print('Independent Features :')
print(x.head())
print('\n Target Variable i.e Survived:')
print(y.head())


# ## Feature scaling 

# In[49]:


min_fare = df['Fare'].min()
max_fare = df['Fare'].max()
df['Fare'] = (df['Fare'] - min_fare) / (max_fare - min_fare)

print(df['Fare'].head())


# In[51]:


mean_Fare = df['Fare'].mean()
std_dev_Fare = df['Fare'].std()
df['Fare'] =(df['Fare'] - mean_Fare) / std_dev_Fare

print(df['Fare'].head())


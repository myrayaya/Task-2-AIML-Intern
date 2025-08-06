import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading in the dataset
df = pd.read_csv(r'D:/all my games/programming vscode/archive/Titanic-Dataset.csv')

#preview of first few rows
df.head()

#inspecting the data structure

#checking shape of the dataset
print(df.shape)

#viewing column names and datatypes
print(df.info())

#checking the no. of missing values per column
print(df.isnull().sum())

# GENERATING STATISTICS

#descriptive statistics for numeric columns
df.describe()

#summary for all columns
df.describe(include = 'all')

#handling missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# CREATING HISTOGRAMS AND BOXPLOTS

#histogram
numeric_cols = df.select_dtypes(include = np.number).columns.tolist()
plt.suptitle('Histograms of Numeric Features')
plt.show()

#boxplots
for col in numeric_cols:
    plt.figure(figsize = (6, 1.5))
    sns.boxplot(x = df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# CREATING PAIRPLOTS AND CORRELATION MATRIX TO SHOW RELATIONSHIPS

#pairplot (scatterplot matrix)
sns.pairplot(df[numeric_cols])
plt.suptitle('Pairplot', y = 1.02)
plt.show()

#correlation matrix 
corr = df[numeric_cols].corr()
plt.figure(figsize = (10, 8))
sns.heatmap(corr, annot = True, cmap = 'coolwarm')
plt.title ('Correlation Heatmap')
plt.show()

# FEATURE-LEVEL INSIGHTS 
categorical_cols = df.select_dtypes(include = 'object').columns.tolist()

#counterplot for categorical columns
for col in categorical_cols:
    plt.figure(figsize = (6, 4))
    sns.countplot(x = col, data = df)
    plt.title(f'Distribution of {col}')
    plt.show()


import seaborn as sns
import pandas as pd
import numpy as np

df = sns.load_dataset('titanic')
df = df.dropna()
df.head()

# 1
df.groupby('sex')['age'].sum()[1]
df.groupby('sex')['age'].sum()[1] - df.groupby('sex')['age'].sum()[0]

# 2 
df[(df['sex'] == 'male' and (df['age'] >=40 and df['age'] <= 49))]


df_1 = df[df['sex'] == 'male']
df_2 = df_1[(df_1['age'] >= 40) & (df_1['age'] <= 49)]
df_2['fare'].mean()


# 3 .
X = np.array([[2,4],
             [1,7],
             [7,8]])
y = np.array([[10],
             [5],
             [15]])

XtX = np.dot(X.transpose(), X)
XtX_inv = np.linalg.inv(XtX)
a = X.dot(XtX_inv)
b = a.dot(X.transpose())
h = b.dot(y)



# 4.
np.random.seed(2025)
array_2d = np.random.randint(1, 13, 200).reshape((50, 4))
array_2d[:4,:]

array_2d.mean(axis=1)
np.max(array_2d.mean(axis=1))


# 5 각 학생별 변동폭(최고점-최저점)
len(array_2d)
result = []
for i in range(len(array_2d)):
  result.append(np.max(array_2d[i,]) - np.min(array_2d[i,]))
sum(result)


sum(array_2d.max(axis=1) - array_2d.min(axis=1))
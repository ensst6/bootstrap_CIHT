import pandas as pd
import numpy as np
np.random.seed(42)


df = pd.DataFrame()

test_list = []
for _ in range(0,1000):
    test_list.append('success')
for _ in range(1000,2000):
    test_list.append('failure')

df['outcome'] = test_list

dist1 = np.random.normal(0, 0.5, 1000)
dist2 = np.random.normal(1, 0.5, 1000)

dist = np.append(dist1, dist2)

df['numbers'] = dist

print(df.info())
print(df.head())
print(df.tail())

print(df[df['outcome']=='success'].mean())
print(df[df['outcome']=='success'].std())

print(df[df['outcome']=='failure'].mean())
print(df[df['outcome']=='failure'].std())

df.to_csv('nml2.csv', index=False)

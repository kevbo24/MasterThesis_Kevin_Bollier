import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('data/data_processed/df_processed.csv', index_col = 0)

corr = df.corr(method ='pearson')

plt.figure(figsize=(15, 15))
hm = sns.heatmap(corr, annot = True)

hm.set(title = "Correlation matrix\n")

plt.savefig('data/data_processed/df_corr.png')
plt.show()
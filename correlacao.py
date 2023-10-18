import pandas as pd
import numpy 
import math
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:/Users/Yosdan/OneDrive/Glendo/Elvis/database.csv")


dados_elvis=df

#print(dados_vanessa.head(10))


matrix_correlacao_spearman=dados_elvis.corr(method="spearman")
matrix_correlacao_pearson=dados_elvis.corr(method="pearson")

fig, (ax1, ax2) = plt.subplots(ncols=2)


sns.heatmap(matrix_correlacao_pearson, annot=True, vmin=-1, center=0, cmap='vlag', ax=ax1)
sns.heatmap(matrix_correlacao_spearman, annot=True, vmin=-1, center=0, cmap='vlag', ax=ax2)

ax1.set_title("Pearson")
ax2.set_title("Spearman")

plt.show()

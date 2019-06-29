import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


start = time.time()

data = pd.read_csv("data.csv")

data = data.sample(frac=1).reset_index(drop=True)
XLabels = ['TotalT', 'Temp', 'LSR', 'CA', 'Size', 'IsoT', 'HeatT', 'Ramp', 'F_X', 'Ro', 'logRo', 'P']

xforPCA = data[XLabels]
sc = StandardScaler()
scaled = sc.fit_transform(xforPCA)


pca = PCA(n_components=2)
# print(pca.explained_variance_ratio_)

principalComponents = pca.fit_transform(scaled)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC1', 'PC2'])
finalDf = pd.concat([principalDf, data[['ID']]], axis = 1)


dataWithPCA = pd.concat([data, principalDf[['PC1']]], axis = 1)

dataWithPCA = pd.concat([dataWithPCA, principalDf[['PC2']]], axis = 1)

dataWithPCA.to_csv("dataWithPCA.csv", index=False)




principalDf.plot.scatter(x="PC1",y="PC2" )
plt.show()
# print(finalDf['PC1'])

end = time.time()
duration = end - start
print("Execution Time is:", duration /60, "min")
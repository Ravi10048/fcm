from sklearn import datasets
import pandas as pd
import skfuzzy as fuzz 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
iris_data=datasets.load_iris()
X=iris_data['data']
Y=iris_data['target']

print(X)
print(Y)


# Define the number of clusters
n_clusters = 3
 
# Apply fuzzy c-means clustering
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
)
 
# Predict cluster membership for each data point
cluster_membership = np.argmax(u, axis=0)
y=cluster_membership
 
# Print the cluster centers
# cntr=np.reshape(cntr,(-1,1))
print('Cluster Centers:', cntr)
print("cluster membership:",cluster_membership)

#plotting all clusters
plt.figure(figsize=(8,8))
plt.scatter(X[y==0,0],X[y==0,1],s=50,c='green',label="cluster1")
plt.scatter(X[y==1,0],X[y==1,1],s=50,c='red',label="cluster2")
plt.scatter(X[y==2,0],X[y==2,1],s=50,c='yellow',label="cluster3")

#plotting centroid

plt.scatter(cntr[:,0],cntr[:,1],s=100,c='cyan',label='centroid')
plt.title("iris clustering")
plt.show()


















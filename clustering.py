#-------------------------------------------------------------------------
# AUTHOR: Ahhad Mukhtar
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df.values[:, :64])
max_silhouette_score = 0
current_silhouette_score = 0
silhouette_array = [] #add values to store for graph
k = 1
#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
while k < 20:
    k += 1
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
    current_silhouette_score = silhouette_score(X_training, kmeans.labels_)
    silhouette_array.append(current_silhouette_score)
    if current_silhouette_score > max_silhouette_score:
        max_silhouette_score = current_silhouette_score
print("Value of k for highest silhoutte score {} is {}".format(max_silhouette_score, k))

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code
a = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
plt.plot(a, silhouette_array, color = 'red')
plt.xlabel("K value")
plt.ylabel("Silhouette Score")
plt.grid(True)
# plt.gca().set_ylim(0,1)
plt.show()
#reading the test data (clusters) by using Pandas library
#--> add your Python code here
df = pd.read_csv('testing_data.csv', sep=',', header=None)
X_test = np.array(df.values[:, :64])
#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(df.values).reshape(1, 3823)[0]
#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here

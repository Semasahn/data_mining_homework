import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.metrics import silhouette_score


# I loaded the data from a CSV file
# The dataset is in "ch_2/iris.csv" file. I used pandas to read it.
file_path = "ch_2/iris.csv"                                      
df = pd.read_csv(file_path)

# I checked the first 5 rows of the dataset
# This helps to understand how the data looks like.
print("First 5 Rows of the Dataset:\n", df.head())

# I selected the feature columns
# These columns are used for clustering: 'sepal_length', 'sepal_width', 'petal_length', and 'petal_width'.
feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df_features = df[feature_columns]

# I scaled the features using StandardScaler
# Scaling makes the data uniform and helps the algorithm work better.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

# I ran KMeans with different numbers of clusters (1 to 10)
# The goal is to find the best number of clusters for the data.
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# I created an elbow graph to find the optimal number of clusters
# The 'elbow' shows where the inertia decreases slowly, which helps find the best K.
plt.figure()
plt.plot(K, inertia, 'bo-', marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# I chose the optimal number of clusters (3) based on the elbow method
optimal_k = 3  
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_data)

# I calculated the silhouette score to measure how well the clusters are separated
# Silhouette score tells how similar the points are within a cluster and how different the clusters are.
silhouette_avg = silhouette_score(scaled_data, df['cluster'])
print(f"Silhouette Score: {silhouette_avg}")

# I visualized the clusters on a scatter plot
# The colors show which cluster each point belongs to, and the red crosses are the cluster centers.
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    scaled_data[:, 0], scaled_data[:, 1],
    c=df['cluster'], cmap='viridis', s=50
)
plt.scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    c='red', marker='x', s=100, label='Cluster Centers'
)

# I added a legend to the graph to label the clusters
# The legend helps to understand which color corresponds to which cluster.
cmap = plt.cm.get_cmap('viridis', optimal_k)  
labels = [f"{i}" for i in range(optimal_k)]  
handles = [mpatches.Patch(color=cmap(i / optimal_k), label=labels[i]) for i in range(len(labels))]

plt.legend(
    handles=handles,
    title="Clusters",
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

# I set the title and axis labels for the plot
plt.title("K-Means Clusters")
plt.xlabel("Scaled Feature 1")
plt.ylabel("Scaled Feature 2")
plt.tight_layout()
plt.show()

# I printed the number of samples in each cluster
# This tells how many data points are in each cluster.
print("\nNumber of Samples in Each Cluster:\n", df['cluster'].value_counts())

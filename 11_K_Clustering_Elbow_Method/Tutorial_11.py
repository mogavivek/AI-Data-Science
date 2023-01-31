from sklearn import datasets
import pandas as pd
from matplotlib import pyplot as plt

# This topic new library used
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

iris = datasets.load_iris()
iris.feature_names

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df_new = df.drop((['sepal length (cm)','sepal width (cm)']), axis='columns')
print(df_new.head())

plt.scatter(df_new['petal length (cm)'], df_new['petal width (cm)'])
# plt.show()

# Now importing our topic K clustering with 4 number
km = KMeans(n_clusters=4)
print(km)

# Predicting the values
y_predicted = km.fit_predict(df_new[['petal length (cm)','petal width (cm)']])
print('\n',y_predicted)

df_new['cluster'] = y_predicted
print('\n', df_new.head())

# Now plotting our all cluster together,

df_new0 = df_new[df_new.cluster==0]
df_new1 = df_new[df_new.cluster==1]
df_new2 = df_new[df_new.cluster==2]
df_new3 = df_new[df_new.cluster==3]

plt.scatter(df_new0['petal length (cm)'],df_new0['petal width (cm)'], color='green')
plt.scatter(df_new1['petal length (cm)'],df_new1['petal width (cm)'], color='red')
plt.scatter(df_new2['petal length (cm)'],df_new2['petal width (cm)'], color='blue')
plt.scatter(df_new3['petal length (cm)'],df_new3['petal width (cm)'], color='black')

plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
# plt.show()

# Now applying MinMaxScaler method,
# It will set the value between 0 to 1
scaler = MinMaxScaler()
scaler.fit(df_new[['petal width (cm)']])
df_new['petal width (cm)'] = scaler.transform(df_new[['petal width (cm)']])

scaler.fit(df_new[['petal length (cm)']])
df_new['petal length (cm)'] = scaler.transform(df_new[['petal length (cm)']])
print(df_new.head())

# New kluster and scatter plot
km = KMeans(n_clusters=4)
y_predicted = km.fit_transform(df[['petal length (cm)', 'petal width (cm)']])

df_new['cluster'] = y_predicted
print('\n', df_new)

# Applying the centroid at each and every clusters
print('\n', km.cluster_centers_)

df_new0 = df_new[df_new.cluster==0]
df_new1 = df_new[df_new.cluster==1]
df_new2 = df_new[df_new.cluster==2]
df_new3 = df_new[df_new.cluster==3]

plt.scatter(df_new0['petal length (cm)'],df_new0['petal width (cm)'], color='green', label='petal width (cm)')
plt.scatter(df_new1['petal length (cm)'],df_new1['petal width (cm)'], color='red', label='petal width (cm)')
plt.scatter(df_new2['petal length (cm)'],df_new2['petal width (cm)'], color='blue', label='petal width (cm)')
plt.scatter(df_new3['petal length (cm)'],df_new3['petal width (cm)'], color='black', label='petal width (cm)')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='Purple', marker='*',label='centroid')

plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.legend()
plt.show()


# Now applying the elbow technique
k_rng = range(1,10) # The value depends on requirement
sse = []            # Storing the sum of square error is in array

for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df_new[['petal length (cm)','petal width (cm)']])
    sse.append(km.inertia_)

print("\nSum of squared error values")
print(sse)


plt.xlabel('k')
plt.ylabel('_Sum of squared error')
plt.plot(k_rng,sse)
plt.show()
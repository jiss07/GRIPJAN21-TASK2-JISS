
# Author: JISS PETER , Data Science Intern @ The Spark Foundation
The task related dataset is available on the web url https://bit.ly/3kXTdox. This dataset can be downloaded locally or can access directly in the code. Since iris dataset is a wellknown dats set available directly in the python library sklearn.datasets, I have imported the same directly in to my code.

## **Data Science & Business Analytics Internship at The Sparks Foundation.**

#GRIPJAN21
## GRIP-Task2 - Prediction using Unsupervised ML

### **Task Description**
In this clustering task related to Unsupervised learning we will try to predict the optimum number of clusters
I have used python and its built in libraries such as numpy, pandas, kmeans clustering techniques as well as used matplot lib library for ploting the predicted results as well as the variable ploting.

#### The intent of this task related coding work is 
#### 1) To run the unsupervised learning techniques on the given 'iris' dataset and to predict the optimum number of clusters and
####  2) Represent those clusters visually by ploting against their centroids


```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df.head() # See the first 5 rows
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#To see how many rows and columns does the data contain.
iris_df.shape
```




    (150, 4)



### 1) Now we need to find out the optimum number of clusters for K Means.
#### The first step towards the same is to determine the correct value of K


```python
# Finding the optimum number of clusters for k-means classification

x = iris_df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()
```


![png](GRIP-Task2-PredictionUsingUnsupervisedML_files/GRIP-Task2-PredictionUsingUnsupervisedML_5_0.png)


## The above graph is having an elbow shape and there by its being called as "Elbow Method" and the optimum clusters is where the elbow occurs.

### This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration. In this case it is happening at the value of 3

#### We do choose number of optimum cluster as 3 , which answers the first portion of our initial problem statement

# Applying kmeans to the dataset / Creating the kmeans classifier


```python
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
```


```python
# Visualising the clusters - On the first two columns
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'blue', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'yellow', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'black', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'red', label = 'Centroids')

plt.legend()
```




    <matplotlib.legend.Legend at 0x1dcca304e10>




![png](GRIP-Task2-PredictionUsingUnsupervisedML_files/GRIP-Task2-PredictionUsingUnsupervisedML_8_1.png)


# Conclusion
### Inorder to complete this GRIP-Task 2 , we have figured out the optimum number of clusters for the given iris dataset as 3 and also the clusters have been represented visually as a scattered plot has been above as expected in the task related problem statement.
## Completed Task 2.
### Thank you for going through this solution


```python

```

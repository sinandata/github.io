---
layout: post
title: "PCA on Wine Quality Dataset"
subtitle: "Principal Component Analysis for unsupervised learning"
tags: [Unsupervised learning, PCA]
comments: true
---

We will use the Wine Quality Data Set for red wines created by P. Cortez et al. It has 11 variables and 1600 observations.

**Data science problem:** Find out which features of wine are important to determine its quality.

We will use the Wine Quality Data Set for red wines created by P. Cortez et al. It has 11 variables and 1600 observations.

Steps to be taken from a _data science_ perspective:

1. [Set the research goal:](#1-research-goal) We want to explain what properties of wine define the quality.
2. [Acquire data:](#2-acquire-data) We will download the data set from a repository.
3. [Prepare data:](#3-prepare-data) We will prepare data for the analysis.
4. [Build model:](#4-model-selection) Build machine learning model you want to use for data analysis.
- [Initial inspection](#a-initial-inspection)
- [Interpreting the results](#b-interpreting-the-results)
- [Predictive model](#c-predictive-model)

- [Resources](#resources)
- [Link to GitHub repo](#link-to-github-repo)

## 1. Research goal

We want to investigate what properties of wine define its quality.

We also want to see which of those properties are required to predict the quality of wine. This way we can safely ignore the unnecessary information next time we collect new data.

## 2. Acquire data


```python
import numpy as np
import pandas as pd

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

data = pd.read_csv(url, sep=";")
```

Let's see the first few rows of the dataset:


```python
data.head()
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Prepare data

The last column is the target variable. We will use the rest as predictor variables. Slice from first column to one before the last.


```python
X = data.loc[:,'fixed acidity':'alcohol']
y = data['quality']
```

Double check:


```python
X.columns # X is a pandas data frame
```




    Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol'],
          dtype='object')




```python
y.name # y is a pandas series
```




    'quality'



Before performing PCA, the dataset has to be standardized (i.e. subtracting mean, dividing by the standard deviation) The scikit-learn PCA package probably performs this internally, but we will do it anyway.


```python
from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X)
scaler
```




    StandardScaler(copy=True, with_mean=True, with_std=True)



We can see the mean and STD of each variable used for standardization:


```python
print('Mean of each variable:')
print(scaler.mean_)
print('\nStd of each variable:')
print(scaler.scale_)
```

    Mean of each variable:
    [ 8.31963727  0.52782051  0.27097561  2.5388055   0.08746654 15.87492183
     46.46779237  0.99674668  3.3111132   0.65814884 10.42298311]

    Std of each variable:
    [1.74055180e+00 1.79003704e-01 1.94740214e-01 1.40948711e+00
     4.70505826e-02 1.04568856e+01 3.28850367e+01 1.88674370e-03
     1.54338181e-01 1.69453967e-01 1.06533430e+00]


Perform transformation:


```python
X = scaler.transform(X)
```

## 4. Model selection

### a. Initial inspection

We want to use PCA and take a closer look at the latent variables.


```python
from sklearn.decomposition import PCA

pca = PCA() # creates an instance of PCA class
results = pca.fit(X) # applies PCA on predictor variables
Z = results.transform(X) # create a new array of latent variables
```


```python
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()


plt.plot(results.explained_variance_ratio_*100) # scree plot
plt.show()
```


![png](/images/pca-wine/scree-plot.png)


The above is called a __scree plot__. It shows the variances explained by each latent variable. The first component explains approx. 28% of the variance in the whole dataset.

Ideally, we would like to see an __elbow shape__ in order to decide which PCs to keep and which ones to disregard. In practice, this rarely happens. Most of the time, we use enough PCs so that they explain 95% or 99% of the variation in the data.

By examining the above figure, we can conclude that first 6 variables contain most of the information inside the data.

### b. Interpreting the results

Once we apply the PCA, we are no longer in our familiar domain. We are in a different domain in which the latents are the linear combinations of the original variables, but they don't represent any meaningful properties. Thus, it is impossible to interpret them by themselves.

We usually look at the correlation between the latent variable and original variables. If any of the original variables correlate well with the first few PCs, we usually conclude that the PCs are mainly influenced by the said variables, thus they must be the important ones.

The other approach is that we look at the PCA coefficients. These coefficients tell us how much of the original variables are used in creating the PCs. The higher the coefficient, the more important is the related variable.

Let's put the component (PCA coefficients) into a data frame to see more comfortably:


```python
pd.DataFrame(results.components_)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.489314</td>
      <td>-0.238584</td>
      <td>0.463632</td>
      <td>0.146107</td>
      <td>0.212247</td>
      <td>-0.036158</td>
      <td>0.023575</td>
      <td>0.395353</td>
      <td>-0.438520</td>
      <td>0.242921</td>
      <td>-0.113232</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.110503</td>
      <td>0.274930</td>
      <td>-0.151791</td>
      <td>0.272080</td>
      <td>0.148052</td>
      <td>0.513567</td>
      <td>0.569487</td>
      <td>0.233575</td>
      <td>0.006711</td>
      <td>-0.037554</td>
      <td>-0.386181</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.123302</td>
      <td>-0.449963</td>
      <td>0.238247</td>
      <td>0.101283</td>
      <td>-0.092614</td>
      <td>0.428793</td>
      <td>0.322415</td>
      <td>-0.338871</td>
      <td>0.057697</td>
      <td>0.279786</td>
      <td>0.471673</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.229617</td>
      <td>0.078960</td>
      <td>-0.079418</td>
      <td>-0.372793</td>
      <td>0.666195</td>
      <td>-0.043538</td>
      <td>-0.034577</td>
      <td>-0.174500</td>
      <td>-0.003788</td>
      <td>0.550872</td>
      <td>-0.122181</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.082614</td>
      <td>0.218735</td>
      <td>-0.058573</td>
      <td>0.732144</td>
      <td>0.246501</td>
      <td>-0.159152</td>
      <td>-0.222465</td>
      <td>0.157077</td>
      <td>0.267530</td>
      <td>0.225962</td>
      <td>0.350681</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.101479</td>
      <td>0.411449</td>
      <td>0.069593</td>
      <td>0.049156</td>
      <td>0.304339</td>
      <td>-0.014000</td>
      <td>0.136308</td>
      <td>-0.391152</td>
      <td>-0.522116</td>
      <td>-0.381263</td>
      <td>0.361645</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.350227</td>
      <td>-0.533735</td>
      <td>0.105497</td>
      <td>0.290663</td>
      <td>0.370413</td>
      <td>-0.116596</td>
      <td>-0.093662</td>
      <td>-0.170481</td>
      <td>-0.025138</td>
      <td>-0.447469</td>
      <td>-0.327651</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.177595</td>
      <td>-0.078775</td>
      <td>-0.377516</td>
      <td>0.299845</td>
      <td>-0.357009</td>
      <td>-0.204781</td>
      <td>0.019036</td>
      <td>-0.239223</td>
      <td>-0.561391</td>
      <td>0.374604</td>
      <td>-0.217626</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.194021</td>
      <td>0.129110</td>
      <td>0.381450</td>
      <td>-0.007523</td>
      <td>-0.111339</td>
      <td>-0.635405</td>
      <td>0.592116</td>
      <td>-0.020719</td>
      <td>0.167746</td>
      <td>0.058367</td>
      <td>-0.037603</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.249523</td>
      <td>0.365925</td>
      <td>0.621677</td>
      <td>0.092872</td>
      <td>-0.217671</td>
      <td>0.248483</td>
      <td>-0.370750</td>
      <td>-0.239990</td>
      <td>-0.010970</td>
      <td>0.112320</td>
      <td>-0.303015</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.639691</td>
      <td>0.002389</td>
      <td>-0.070910</td>
      <td>0.184030</td>
      <td>0.053065</td>
      <td>-0.051421</td>
      <td>0.068702</td>
      <td>-0.567332</td>
      <td>0.340711</td>
      <td>0.069555</td>
      <td>-0.314526</td>
    </tr>
  </tbody>
</table>
</div>



The above is the _coefficient matrix_. The first row is the coefficients that generated the first PC. In other words, the first PC was generated using the following formula:

PC1 = (fixed acidity * 0.489314) + (volatile acidity * -0.238584) + ... + (alcohol * -0.113232)

If we choose to use the first 6 PCs, what do we call them? This is a bit tricky, because they are wieghted combinations of the original variables. Thus, a wine expert may help us to name them. Let's assume that the expert gave us the following names for the new (latent) variables: 1.Acidity, 2.Sulfides, 3.More alcohol, 4.Chlorides, 5.More residual sugar, 6. Less pH


```python
pd.DataFrame(Z[:,:6], columns=list(
[u'Acidity', u'Sulfides', u'More alcohol', u'Chlorides', u'More residual sugar', u'Less pH'])).head(10)
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
      <th>Acidity</th>
      <th>Sulfides</th>
      <th>More alcohol</th>
      <th>Chlorides</th>
      <th>More residual sugar</th>
      <th>Less pH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.619530</td>
      <td>0.450950</td>
      <td>-1.774454</td>
      <td>0.043740</td>
      <td>0.067014</td>
      <td>-0.913921</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.799170</td>
      <td>1.856553</td>
      <td>-0.911690</td>
      <td>0.548066</td>
      <td>-0.018392</td>
      <td>0.929714</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.748479</td>
      <td>0.882039</td>
      <td>-1.171394</td>
      <td>0.411021</td>
      <td>-0.043531</td>
      <td>0.401473</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.357673</td>
      <td>-0.269976</td>
      <td>0.243489</td>
      <td>-0.928450</td>
      <td>-1.499149</td>
      <td>-0.131017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.619530</td>
      <td>0.450950</td>
      <td>-1.774454</td>
      <td>0.043740</td>
      <td>0.067014</td>
      <td>-0.913921</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.583707</td>
      <td>0.569195</td>
      <td>-1.538286</td>
      <td>0.023750</td>
      <td>-0.110076</td>
      <td>-0.993626</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-1.101464</td>
      <td>0.608015</td>
      <td>-1.075915</td>
      <td>-0.343959</td>
      <td>-1.133382</td>
      <td>0.175000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-2.248708</td>
      <td>-0.416835</td>
      <td>-0.986837</td>
      <td>-0.001203</td>
      <td>-0.780435</td>
      <td>0.286057</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-1.086887</td>
      <td>-0.308569</td>
      <td>-1.518150</td>
      <td>0.003315</td>
      <td>-0.226727</td>
      <td>-0.512634</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.654790</td>
      <td>1.665207</td>
      <td>1.209476</td>
      <td>-0.824635</td>
      <td>1.718501</td>
      <td>-0.476497</td>
    </tr>
  </tbody>
</table>
</div>



__Usually, naming the new variables is dangerous unless we confidently conclude that the latent variables represent known properties. This requires expert knowledge.__

### c. Predictive model

We can compare the prediction powers of the original variables and latent variables. Since the target is a categorical variable, we will perform a classification operation. We will use KNN for this purpose.

__1. Using the original data set__


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X,y)
pred = neigh.predict(X)
print('Confusion matrix:')
print(confusion_matrix(pred,y))
print('\nAccuracy:')
print(accuracy_score(pred,y))
```

    Confusion matrix:
    [[  6   5   2   0   0   0]
     [  0  18  10  13   0   0]
     [  2  17 589  87  18   2]
     [  2  12  75 511  36   7]
     [  0   1   4  27 144   7]
     [  0   0   1   0   1   2]]

    Accuracy:
    0.7942464040025016


__2. Using the first 6 PCs__


```python
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(Z[:,:6],y)
pred = neigh.predict(Z[:,:6])
print('Confusion matrix:')
print(confusion_matrix(pred,y))
print('\nAccuracy:')
print(accuracy_score(pred,y))
```

    Confusion matrix:
    [[  7   4   5   1   0   0]
     [  0  18  11  11   3   0]
     [  2  16 584  92  26   4]
     [  1  14  75 507  37   6]
     [  0   1   6  27 132   5]
     [  0   0   0   0   1   3]]

    Accuracy:
    0.7823639774859287


Using 6 variables instead of 11, we achive almost the same accuracy in our prediction.

Note: Here, we used the training set for prediction. Ideally, we would want to use two separate sets for training the model and testing it.

## Resources

* D. Cielen, A. Meysman, M. Ali, Introducing Data Science: Big Data Machine Learning and more using Python tools, Manning Publications, 2016.
* Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
* P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

## Link to GitHub repo

You can download the Notebook from my [GitHub page](https://github.com/goksinan/pca_on_wine_quality_data.git)

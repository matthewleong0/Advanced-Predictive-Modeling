# <p style="text-align: center;">MIS 382N: ADVANCED PREDICTIVE MODELING - MSBA</p>
# <p style="text-align: center;">Assignment 4</p>
## <p style="text-align: center;">Total points: 80 </p>
## <p style="text-align: center;">Due: November 9, submitted via Canvas by 11:59 pm</p>

Your homework should be written in a **Jupyter notebook**. You may work in groups of two if you wish. Your partner needs to be from the same section. Only one student per team needs to submit the assignment on Canvas.  But be sure to include name and UTEID for both students.  Homework groups will be created and managed through Canvas, so please do not arbitrarily change your homework group. If you do change, let the TA know. 

Also, please make sure your code runs and the graphics (and anything else) are displayed in your notebook before submitting. (%matplotlib inline)

### Name(s)
1. Matthew Leong
2. Chirag Ramesh

### My Contribution:

I focused on the decision tree and classification report questions (2,4). Chirag handled the other ones and we both worked on peer reviewing our class supplements together.

# Question 1 (20 pts) - Principal Component Analysis

Download dataset from [this link](https://drive.google.com/file/d/1U6FDSQQSUzvw3Ygtgp8I5ea0-dL-orSE/view?usp=sharing).


In this problem we will be applying PCA on the Lending Club loan dataset. A simplified version of the dataset with reduced number of samples. Please use the dataset in the above link. We will use reduced number of features and only two classes as shown below.


```python
import os, sys, re
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the csv file and fill Nan/empty values as 0
dataset = pd.read_csv('loan.csv')
df = dataset.fillna(0)

# We will be using only two classes and group them as below
def LoanResult(status):
    if (status == 'Fully Paid') or (status == 'Current'):
        return 1
    else:
        return 0

df['loan_status'] = df['loan_status'].apply(LoanResult)

# Set of features which indicate the dimensionality of the data
df = df[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
             'emp_length', 'home_ownership','annual_inc', 'verification_status', 'loan_status',
             'purpose','addr_state', 'dti','open_acc', 'pub_rec', 'revol_bal', 'revol_util', 
             'initial_list_status', 'recoveries','collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt',
             'application_type', 'tot_coll_amt', 'tot_cur_bal', 'avg_cur_bal', 'chargeoff_within_12_mths',
             'pub_rec_bankruptcies', 'tax_liens', 'debt_settlement_flag']]

#For simplicity, in this question, we select all columns that do not contain integer of float type of data. Then, one hot encoding is performed.
df_cat = df.select_dtypes(exclude=['int64', 'float64'])
df = pd.get_dummies(df, df_cat.columns.values)

df.shape

# We select the `loan_status` column as the target column.  
```

    C:\Users\mattl\anaconda3\lib\site-packages\IPython\core\interactiveshell.py:3145: DtypeWarning: Columns (123,124,125,128,129,130,133,139,140,141) have mixed types.Specify dtype option on import or set low_memory=False.
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
    




    (226067, 161)



Use Principal Component Analysis (PCA) to solve this problem.  

* **1.1 (1 pt)** Perform the following steps to prepare the dataset:

    * Select the 'loan_status' column as the target column and the rest of the columns from the dataframe df as X. 

    * Split the dataset into train and test set with 25% data in test set and random_state = 42

    * Perform [Min-Max Scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) on the dataset. Remember that when we have training and testing data, we fit preprocessing parameters on training data and apply them to all testing data. You should scale only the features (independent variables), not the target variable y.
    
    Note: X should have 160 features.
    

* **1.2 (8 pts)** Use [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) and reduce the dimension of X to the following components: 100, 30, 10. For each of the three models, print the percentage(%) of variance captured by each of the compnonets and plot the scree plot the [scree plot]
(https://www.kindsonthegenius.com/principal-components-analysispca-in-python-step-by-step/).  (PCA should be fit only on X_train).


* **1.3 (5 pts)** Train LogisticRegression(random_state=4,max_iter=10000) with the full dimension X and each of the redued dimension X from PCA in the previous step (100, 30 and 10 dimensions). Print the classification reports for all the models -  this will print the class-wise Precision, Recall and F1 score. More details on classification report can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) (Note: you will be training logistic regression 4 times (160, 100, 30 and 10 dimensional X) and will print 4 classification reports)


* **1.4 (4 pts)** [Plot](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_roc_curve_visualization_api.html#sphx-glr-auto-examples-miscellaneous-plot-roc-curve-visualization-api-py) the ROC curves for all models (trained using dataset containing all dimensions and dataset containing reduced dimensions, total 4 models). ROC curve is used to study the classifier's output. Details on ROC can be found [here](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html).


* **1.5 (2 pts)** What do you observe from the ROC curves? 

# ANSWER

### 1.1


```python
#set Xs and Y, split data
y = df['loan_status'].values
X = df.drop(['loan_status'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=42)

#Performing min-max scaling
from sklearn.preprocessing import MinMaxScaler
X_scaler = MinMaxScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)
X # we can see that X has 160 features

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
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>...</th>
      <th>last_pymnt_d_Feb-2019</th>
      <th>last_pymnt_d_Jan-2019</th>
      <th>last_pymnt_d_Jul-2018</th>
      <th>last_pymnt_d_Nov-2018</th>
      <th>last_pymnt_d_Oct-2018</th>
      <th>last_pymnt_d_Sep-2018</th>
      <th>application_type_Individual</th>
      <th>application_type_Joint App</th>
      <th>debt_settlement_flag_N</th>
      <th>debt_settlement_flag_Y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2500</td>
      <td>2500</td>
      <td>2500.0</td>
      <td>13.56</td>
      <td>84.92</td>
      <td>55000.0</td>
      <td>18.24</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>4341</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30000</td>
      <td>30000</td>
      <td>30000.0</td>
      <td>18.94</td>
      <td>777.23</td>
      <td>90000.0</td>
      <td>26.52</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>12315</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5000</td>
      <td>5000</td>
      <td>5000.0</td>
      <td>17.97</td>
      <td>180.69</td>
      <td>59280.0</td>
      <td>10.51</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>4599</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4000</td>
      <td>4000</td>
      <td>4000.0</td>
      <td>18.94</td>
      <td>146.51</td>
      <td>92000.0</td>
      <td>16.74</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>5468</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30000</td>
      <td>30000</td>
      <td>30000.0</td>
      <td>16.14</td>
      <td>731.78</td>
      <td>57250.0</td>
      <td>26.35</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>829</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>226062</th>
      <td>20000</td>
      <td>20000</td>
      <td>20000.0</td>
      <td>22.35</td>
      <td>767.44</td>
      <td>55000.0</td>
      <td>20.49</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>14442</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>226063</th>
      <td>10000</td>
      <td>10000</td>
      <td>10000.0</td>
      <td>6.67</td>
      <td>307.27</td>
      <td>52000.0</td>
      <td>24.65</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>19319</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>226064</th>
      <td>13000</td>
      <td>13000</td>
      <td>13000.0</td>
      <td>7.21</td>
      <td>402.66</td>
      <td>90000.0</td>
      <td>4.08</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>18394</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>226065</th>
      <td>10000</td>
      <td>10000</td>
      <td>10000.0</td>
      <td>18.94</td>
      <td>366.26</td>
      <td>33280.0</td>
      <td>31.61</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>17177</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>226066</th>
      <td>15000</td>
      <td>15000</td>
      <td>15000.0</td>
      <td>10.08</td>
      <td>319.30</td>
      <td>90000.0</td>
      <td>9.56</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>10840</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>226067 rows × 160 columns</p>
</div>



### 1.2


```python
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

PCA_100= PCA(n_components=100)
PCA_100_train = PCA_100.fit_transform(X_train)
PCA_100_test = PCA_100.transform(X_test)
percent_variance = np.round(PCA_100.explained_variance_ratio_* 100, decimals =2)
plt.bar(x= range(1,101), height=percent_variance)
plt.ylabel('Percentate of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA 100 Scree Plot')
plt.show()
print("The percentage of variance captured by each component is \n",PCA_100.explained_variance_ratio_*100)

```


    
![png](output_11_0.png)
    


    The percentage of variance captured by each component is 
     [9.17082119 6.99971575 6.02589021 5.48989425 4.92126831 4.59775726
     3.85181561 3.38804923 3.2150326  2.88061821 2.81953987 2.29388169
     2.2207015  1.81506747 1.57670488 1.49458232 1.26754512 1.2262638
     1.1802938  1.16388147 1.09824821 1.08200897 1.05308167 1.01445219
     0.97228646 0.96448067 0.94866012 0.92494468 0.91344465 0.84436517
     0.83207147 0.81085962 0.79798218 0.79262794 0.75341884 0.71117759
     0.69874763 0.6946323  0.69082124 0.66287992 0.58470142 0.56386785
     0.53160026 0.50776262 0.49148929 0.48995033 0.48219288 0.44845559
     0.44692678 0.4021699  0.39106379 0.3832751  0.37587354 0.36581935
     0.36368889 0.35514385 0.34676433 0.33472889 0.32012344 0.30835358
     0.27886389 0.25902673 0.24722146 0.24525374 0.24335679 0.2419016
     0.23329124 0.22715652 0.20685638 0.20465513 0.1966822  0.18560974
     0.16978367 0.165289   0.16117032 0.14984591 0.14572035 0.14411213
     0.14167096 0.13657597 0.13110187 0.12917168 0.12786669 0.12089028
     0.11948656 0.10826233 0.10314328 0.10075511 0.10012312 0.09982086
     0.09412631 0.0927336  0.08627804 0.0843964  0.08004988 0.07613323
     0.07390217 0.07119277 0.0700894  0.06195944]
    


```python
PCA_30= PCA(n_components=30)
PCA_30_train = PCA_30.fit_transform(X_train)
PCA_30_test = PCA_30.transform(X_test)
percent_variance = np.round(PCA_30.explained_variance_ratio_* 100, decimals =2)
plt.bar(x= range(1,31), height=percent_variance)
plt.ylabel('Percentate of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA 30 Scree Plot')
plt.show()
print("The percentage of variance captured by each component is \n",PCA_30.explained_variance_ratio_*100)

```


    
![png](output_12_0.png)
    


    The percentage of variance captured by each component is 
     [9.17082119 6.99971575 6.0258902  5.48989423 4.9212682  4.59775723
     3.85181548 3.38804884 3.21503178 2.88061656 2.8195366  2.29386953
     2.22069294 1.81500383 1.57659504 1.49422797 1.26683515 1.22423519
     1.17884798 1.16265047 1.0957024  1.08143424 1.05194416 1.00230886
     0.96773001 0.96302523 0.9367984  0.92246332 0.90772357 0.83406599]
    


```python
PCA_10= PCA(n_components=10)
PCA_10_train = PCA_10.fit_transform(X_train)
PCA_10_test = PCA_10.transform(X_test)
percent_variance = np.round(PCA_10.explained_variance_ratio_* 100, decimals =2)
plt.bar(x= range(1,11), height=percent_variance)
plt.ylabel('Percentate of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA 10 Scree Plot')
plt.show()
print("The percentage of variance captured by each component is \n",PCA_10.explained_variance_ratio_*100)

```


    
![png](output_13_0.png)
    


    The percentage of variance captured by each component is 
     [9.17082119 6.99971575 6.02589021 5.48989425 4.92126831 4.59775726
     3.85181559 3.38804915 3.21503216 2.88061741]
    

### 1.3


```python
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

Full160 = LogisticRegression(random_state=4,max_iter=10000).fit(X_train, y_train)
y_predict = Full160.predict(X_test)
print(classification_report(y_test, y_predict, labels=[0,1])) 

```

                  precision    recall  f1-score   support
    
               0       0.76      0.46      0.57       850
               1       0.99      1.00      0.99     55667
    
        accuracy                           0.99     56517
       macro avg       0.88      0.73      0.78     56517
    weighted avg       0.99      0.99      0.99     56517
    
    


```python
PCA100_Est = LogisticRegression(random_state=4,max_iter=10000).fit(PCA_100_train, y_train)
y_predict = PCA100_Est.predict(PCA_100_test)
print(classification_report(y_test, y_predict, labels=[0,1]))

```

                  precision    recall  f1-score   support
    
               0       0.71      0.21      0.33       850
               1       0.99      1.00      0.99     55667
    
        accuracy                           0.99     56517
       macro avg       0.85      0.61      0.66     56517
    weighted avg       0.98      0.99      0.98     56517
    
    


```python
PCA30_Est = LogisticRegression(random_state=4,max_iter=10000).fit(PCA_30_train, y_train)
y_predict = PCA30_Est.predict(PCA_30_test)
print(classification_report(y_test, y_predict, labels=[0,1]))
```

                  precision    recall  f1-score   support
    
               0       0.30      0.05      0.08       850
               1       0.99      1.00      0.99     55667
    
        accuracy                           0.98     56517
       macro avg       0.64      0.52      0.54     56517
    weighted avg       0.98      0.98      0.98     56517
    
    


```python
PCA10_Est = LogisticRegression(random_state=4,max_iter=10000).fit(PCA_10_train, y_train)
y_predict = PCA10_Est.predict(PCA_10_test)
print(classification_report(y_test, y_predict, labels=[0,1]))
```

                  precision    recall  f1-score   support
    
               0       0.00      0.00      0.00       850
               1       0.98      1.00      0.99     55667
    
        accuracy                           0.98     56517
       macro avg       0.49      0.50      0.50     56517
    weighted avg       0.97      0.98      0.98     56517
    
    

    C:\Users\mattl\anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1221: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

### 1.4


```python
from sklearn.metrics import plot_roc_curve
ax = plt.gca()
Full160_plot = plot_roc_curve(Full160, X_test, y_test, ax=ax)
PCA100_Est_plot = plot_roc_curve(PCA100_Est, PCA_100_test, y_test, ax=ax)
PCA30_Est_plot = plot_roc_curve(PCA30_Est, PCA_30_test, y_test, ax=ax)
PCA10_Est_plot = plot_roc_curve(PCA10_Est, PCA_10_test, y_test, ax=ax)
plt.title('ROC Curves')
plt.show()

```


    
![png](output_20_0.png)
    


### 1.5


The logistic regression with all of the 160 features, PCA 100, and PCA 30 all seem to have similar a similar ROC curve. However, the PCA 10 model differs from them and has the lowest AUC meaning that it has a lower probability of ranking a random positive example more highly than a random negative example. The PCA 10 model has a high false positive rate and low true positive rate. 

# Question 2 (20 pts)- Decision Tree Classifier
Download dataset from [this link](https://drive.google.com/file/d/1iWh0gF2bXOYSnuq843qLxnFphT1HN-lq/view?usp=sharing).

**Customer Eligibility for Deposits**

We will build a Decision Tree classification model to predict if a customer will subscribe or no (yes/no).


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import datasets
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
%matplotlib inline
```


```python
# Loading the data file
bank=pd.read_csv('bank.csv')
bank.head()
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
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>deposit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2343</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1042</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56</td>
      <td>admin.</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>45</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1467</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>technician</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>1270</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>1389</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>2476</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>579</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>admin.</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>184</td>
      <td>no</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>673</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>



Input variables:
# bank client data:
1 - `age` (numeric)

2 - `job` : type of job (categorical: 'admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')

3 - `marital` : marital status (categorical: 'divorced','married','single'; note: 'divorced' means divorced or widowed)

4 - `education` (categorical: 'primary', 'secondary','tertiary')

5 - `default`: has credit in default? (categorical: 'no','yes','unknown')

6 - `balance`: account balance

7 - `housing`: has housing loan? (categorical: 'no','yes','unknown')

8 - `loan`: has personal loan? (categorical: 'no','yes','unknown')

# related with the last contact of the current campaign:
9 - `contact`: contact communication type (categorical: 'cellular','telephone')

10 - `day_of_month` : 1,2....31

11 - `month`: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')

12 - `duration`: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
13 - `campaign`: number of contacts performed during this campaign and for this client (numeric, includes last contact)

14 - `pdays`: number of days that passed by after the client was last contacted from a previous campaign (numeric; 10000 means client was not previously contacted)

15 - `previous`: number of contacts performed before this campaign and for this client (numeric)

16 - `poutcome`: outcome of the previous marketing campaign (categorical: 'failure','other','success','unknown')

# Output variable (desired target):
17 - `y` - has the client subscribed a term deposit? (binary: 'yes','no')

**All pre-processing is done where categorical variables are converted to numeric values and unnecessary columns are dropped.**


```python
# Make a copy for parsing
bank_data = bank.copy()

# Drop 'contact', as every participant has been contacted. 
bank_data.drop('contact', axis=1, inplace=True)
# Drop 'month' and 'day' as they don't have any intrinsic meaning
bank_data.drop('month', axis=1, inplace=True)
bank_data.drop('day', axis=1, inplace=True)

#Convert categorical values to numeric values
# values for "default" : yes/no
bank_data["default"]
bank_data['default_cat'] = bank_data['default'].map( {'yes':1, 'no':0} )
bank_data.drop('default', axis=1,inplace = True)
# values for "housing" : yes/no
bank_data["housing_cat"]=bank_data['housing'].map({'yes':1, 'no':0})
bank_data.drop('housing', axis=1,inplace = True)
# values for "loan" : yes/no
bank_data["loan_cat"] = bank_data['loan'].map({'yes':1, 'no':0})
bank_data.drop('loan', axis=1, inplace=True)
# values for "deposit" : yes/no
bank_data["deposit_cat"] = bank_data['deposit'].map({'yes':1, 'no':0})
bank_data.drop('deposit', axis=1, inplace=True)

# Convert categorical variables to dummies
bank_data = pd.get_dummies(data=bank_data, columns = ['job', 'marital', 'education', 'poutcome'], \
                                   prefix = ['job', 'marital', 'education', 'poutcome'])

# Convert p_days to a probability value
bank_data['recent_pdays'] = np.where(bank_data['pdays'], 1/bank_data.pdays, 1/bank_data.pdays)
# Drop 'pdays'
bank_data.drop('pdays', axis=1, inplace = True)
```


```python
bank_data.head()
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
      <th>age</th>
      <th>balance</th>
      <th>duration</th>
      <th>campaign</th>
      <th>previous</th>
      <th>default_cat</th>
      <th>housing_cat</th>
      <th>loan_cat</th>
      <th>deposit_cat</th>
      <th>job_admin.</th>
      <th>...</th>
      <th>marital_single</th>
      <th>education_primary</th>
      <th>education_secondary</th>
      <th>education_tertiary</th>
      <th>education_unknown</th>
      <th>poutcome_failure</th>
      <th>poutcome_other</th>
      <th>poutcome_success</th>
      <th>poutcome_unknown</th>
      <th>recent_pdays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>2343</td>
      <td>1042</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56</td>
      <td>45</td>
      <td>1467</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>1270</td>
      <td>1389</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55</td>
      <td>2476</td>
      <td>579</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>184</td>
      <td>673</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
bank_data.deposit_cat
```




    0        1
    1        1
    2        1
    3        1
    4        1
            ..
    11157    0
    11158    0
    11159    0
    11160    0
    11161    0
    Name: deposit_cat, Length: 11162, dtype: int64




```python
# Splitting the data into training and test data with 80:20 ratio with random_state=50.
# Building the data model
# Train-Test split: 20% test data
X = bank_data.drop('deposit_cat', 1)
Y = bank_data.deposit_cat
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 50)
```

a. **(8 pts)** Build a decision tree with depths 2,5,10,20 and max depth using gini and entropy criterion; report the train and test error. Refer [Decisison Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) for more information on how to implement using sklearn.

b. **(2 pts)** Explain how the train and test accuracy varies as we increase the depth of the tree.

c. **(4 pts)** List the most important features for the tree with depth=2 and criterion=gini and plot the tree. Name this tree model as `dt2`.

d. **(6 pts)** Report the accuracy and AUC for the test data and plot the ROC curve using `dt2`.


# Answer 2
## (a)


```python
#Generate trees
gini_tree2 = tree.DecisionTreeClassifier(criterion = 'gini',max_depth = 2)
gini_tree2 = gini_tree2.fit(X_train,Y_train)
entropy_tree2 = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth = 2)
entropy_tree2 = entropy_tree2.fit(X_train,Y_train)

gini_tree5 = tree.DecisionTreeClassifier(criterion = 'gini',max_depth = 5)
gini_tree5 = gini_tree5.fit(X_train,Y_train)
entropy_tree5 = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth = 5)
entropy_tree5 = entropy_tree5.fit(X_train,Y_train)

gini_tree10 = tree.DecisionTreeClassifier(criterion = 'gini',max_depth = 10)
gini_tree10 = gini_tree10.fit(X_train,Y_train)
entropy_tree10 = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth = 10)
entropy_tree10 = entropy_tree10.fit(X_train,Y_train)

gini_tree20 = tree.DecisionTreeClassifier(criterion = 'gini',max_depth = 20)
gini_tree20 = gini_tree20.fit(X_train,Y_train)
entropy_tree20 = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth = 20)
entropy_tree20 = entropy_tree20.fit(X_train,Y_train)


#Generate prediction and display output
gini_tree2_trpred = gini_tree2.predict(X_train)
print("Training Accuracy for gini tree with max depth 2:", metrics.accuracy_score(Y_train,gini_tree2_trpred))
gini_tree2_tepred = gini_tree2.predict(X_test)
print("Testing Accuracy for gini tree with max depth 2:", metrics.accuracy_score(Y_test,gini_tree2_tepred))
entropy_tree2_trpred = entropy_tree2.predict(X_train)
print("Training Accuracy for entropy tree with max depth 2:", metrics.accuracy_score(Y_train,entropy_tree2_trpred))
entropy_tree2_tepred = entropy_tree2.predict(X_test)
print("Testing Accuracy for entropy tree with max depth 2:", metrics.accuracy_score(Y_test,entropy_tree2_tepred))
#empty print statement to make output look neater.
print('')

gini_tree5_trpred = gini_tree5.predict(X_train)
print("Training Accuracy for gini tree with max depth 5:", metrics.accuracy_score(Y_train,gini_tree5_trpred))
gini_tree5_tepred = gini_tree5.predict(X_test)
print("Testing Accuracy for gini tree with max depth 5:", metrics.accuracy_score(Y_test,gini_tree5_tepred))
entropy_tree5_trpred = entropy_tree5.predict(X_train)
print("Training Accuracy for entropy tree with max depth 5:", metrics.accuracy_score(Y_train,entropy_tree5_trpred))
entropy_tree5_tepred = entropy_tree5.predict(X_test)
print("Testing Accuracy for entropy tree with max depth 5:", metrics.accuracy_score(Y_test,entropy_tree5_tepred))
print('')

gini_tree10_trpred = gini_tree10.predict(X_train)
print("Training Accuracy for gini tree with max depth 10:", metrics.accuracy_score(Y_train,gini_tree10_trpred))
gini_tree10_tepred = gini_tree10.predict(X_test)
print("Testing Accuracy for gini tree with max depth 10:", metrics.accuracy_score(Y_test,gini_tree10_tepred))
entropy_tree10_trpred = entropy_tree10.predict(X_train)
print("Training Accuracy for entropy tree with max depth 10:", metrics.accuracy_score(Y_train,entropy_tree10_trpred))
entropy_tree10_tepred = entropy_tree10.predict(X_test)
print("Testing Accuracy for entropy tree with max depth 10:", metrics.accuracy_score(Y_test,entropy_tree10_tepred))
print('')

gini_tree20_trpred = gini_tree20.predict(X_train)
print("Training Accuracy for gini tree with max depth 20:", metrics.accuracy_score(Y_train,gini_tree20_trpred))
gini_tree20_tepred = gini_tree20.predict(X_test)
print("Testing Accuracy for gini tree with max depth 20:", metrics.accuracy_score(Y_test,gini_tree20_tepred))
entropy_tree20_trpred = entropy_tree20.predict(X_train)
print("Training Accuracy for entropy tree with max depth 20:", metrics.accuracy_score(Y_train,entropy_tree20_trpred))
entropy_tree20_tepred = entropy_tree20.predict(X_test)
print("Testing Accuracy for entropy tree with max depth 20:", metrics.accuracy_score(Y_test,entropy_tree20_tepred))
```

    Training Accuracy for gini tree with max depth 2: 0.7285250307985217
    Testing Accuracy for gini tree with max depth 2: 0.7268248992386923
    Training Accuracy for entropy tree with max depth 2: 0.7119498264083324
    Testing Accuracy for entropy tree with max depth 2: 0.7089117778772951
    
    Training Accuracy for gini tree with max depth 5: 0.7976257139657297
    Testing Accuracy for gini tree with max depth 5: 0.7760859829825347
    Training Accuracy for entropy tree with max depth 5: 0.7998656064508903
    Testing Accuracy for entropy tree with max depth 5: 0.7783251231527094
    
    Training Accuracy for gini tree with max depth 10: 0.8634785530294545
    Testing Accuracy for gini tree with max depth 10: 0.7859381997313032
    Training Accuracy for entropy tree with max depth 10: 0.8500391981184903
    Testing Accuracy for entropy tree with max depth 10: 0.7895208240035826
    
    Training Accuracy for gini tree with max depth 20: 0.9840967633553589
    Testing Accuracy for gini tree with max depth 20: 0.7393640841916704
    Training Accuracy for entropy tree with max depth 20: 0.9642737148616867
    Testing Accuracy for entropy tree with max depth 20: 0.7478728168383341
    

## (b)



As we increase the depth of the tree, training accuracy and test accuracy both rise up until depth 10. However as we move on to depth 20, we start overfitting. Although the training accuracy rises, the testing accuracy falls due to the latter reaason.

# (c)


```python
dt2 = gini_tree2
#Make a dataframe to help illustrate output
#As it's a depth two tree, only expect two features to be important.

feature_importances = pd.Series(dt2.feature_importances_, index=X.columns)

#Plot showcases 4.
feature_importances.nlargest(4).plot(kind='barh')

#List out the values to show that the rest are 0.
feature_importances.sort_values(ascending = False)
```




    duration               0.849306
    poutcome_success       0.150694
    recent_pdays           0.000000
    job_self-employed      0.000000
    balance                0.000000
    campaign               0.000000
    previous               0.000000
    default_cat            0.000000
    housing_cat            0.000000
    loan_cat               0.000000
    job_admin.             0.000000
    job_blue-collar        0.000000
    job_entrepreneur       0.000000
    job_housemaid          0.000000
    job_management         0.000000
    job_retired            0.000000
    job_services           0.000000
    poutcome_unknown       0.000000
    job_student            0.000000
    job_technician         0.000000
    job_unemployed         0.000000
    job_unknown            0.000000
    marital_divorced       0.000000
    marital_married        0.000000
    marital_single         0.000000
    education_primary      0.000000
    education_secondary    0.000000
    education_tertiary     0.000000
    education_unknown      0.000000
    poutcome_failure       0.000000
    poutcome_other         0.000000
    age                    0.000000
    dtype: float64




    
![png](output_37_1.png)
    



```python
#Graphviz does not work.
#Elected to visualize using sklearn package instead.

#dt2.classes_
#Looked at dt2.class_ to see how it was classified. 
#First value is 0 meaning no term deposit. Second value of 1 meaning Term Deposit.       
#Entered accordingly in the class_names for the graph.
fig = plt.figure(figsize=(25,20))
tree.plot_tree(dt2, 
               feature_names = X.columns,
               class_names = ["No Term Deposit","Term Deposit"],
                   filled=True)
plt.show()
```


    
![png](output_38_0.png)
    


## (d)


```python
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_roc_curve
```


```python
#Did it for practice.
#metrics.confusion_matrix(Y_test, gini_tree2_tepred)
#fpr_tree, tpr_tree, thresholds_tree = metrics.roc_curve(Y_test, gini_tree2_tepred)
```


```python
plot_roc_curve(dt2, X_test, Y_test, alpha=0.8)
plt.title('ROC Curve for dt2')
plt.show()
```


    
![png](output_42_0.png)
    



```python
print("The accuracy for the test data is", metrics.accuracy_score(Y_test,gini_tree2_tepred))
```

    The accuracy for the test data is 0.7268248992386923
    

# Question 3 (15 pts) - Pipeline Implementation from Sklearn 

In this question we will build a pipeline to streamline the ML Workflow. Instead of writing code for each logic, pipelines allow to make modeling easy by removing repeated operations. Generally we define the structure of the pipeline, to include the following steps, data pre-processing, feature selection, model building. 

**Part A (5 pts)**

 * Load the dataset from "vehicle.csv". The target variable is denoted by column_name = 'class'. Print the label class,and perform [LabelEncoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) on them using sklearn.preprocessing.LabelEncoder.

 * Seperate the dataset into features,labels. Split the dataset into train and test set with 20% data in test set and random_state = 50

**Part B (5 pts)**

Now, we will build [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) with the following steps. Remeber to fit the pipeline on training set and predict on test set. Finally report the classification accuracy on the test set.

 * Data Pre-processing : Using SimpleImputer with strategy = 'mean'
 * Standardization : Standardize features by removing the mean and scaling to unit variance using StandardScaler()
 * Model : Use DecisionTreeClassifier with default values
  

**Part C (5 pts)**
  
In part C, we will build on top of the previous part B. In addition to the three steps we will add PCA to the pipeline and use [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to find the best parameters for PCA(number of components) and DecisionTree(max_depth).
  
 * Data Pre-processing : Using SimpleImputer with strategy = 'mean'
 * Standardization : Standardize features by removing the mean and scaling to unit variance using StandardScaler()
 * PCA : Use PCA()
 * Model : Use DecisionTreeClassifier with default values
  
We will now use [GridSearchCV] with default values(https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to estimate best parameters for pca(n_components = [5,8,10]) and Decision Tree (max_depth = [3,5,15]) using the pipeline designed. 

* **Hint:** The `param_grid` argument to gridsearch will be given as `{param_grid={'clf__max_depth': [3, 5, 15], 'pca__n_components': [5, 8, 10]}}`. 

Print the best parameters found by gridsearch. Finally, update the best parameters to the pipeline using pipe_pca.set_params("best params from gridsearchCV"), and report the classification accuracy on test set.

              

## Part A


```python
from sklearn import preprocessing
df=pd.read_csv('vehicle.csv')
df
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
      <th>compactness</th>
      <th>circularity</th>
      <th>distance_circularity</th>
      <th>radius_ratio</th>
      <th>pr.axis_aspect_ratio</th>
      <th>max.length_aspect_ratio</th>
      <th>scatter_ratio</th>
      <th>elongatedness</th>
      <th>pr.axis_rectangularity</th>
      <th>max.length_rectangularity</th>
      <th>scaled_variance</th>
      <th>scaled_variance.1</th>
      <th>scaled_radius_of_gyration</th>
      <th>scaled_radius_of_gyration.1</th>
      <th>skewness_about</th>
      <th>skewness_about.1</th>
      <th>skewness_about.2</th>
      <th>hollows_ratio</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>95</td>
      <td>48.0</td>
      <td>83.0</td>
      <td>178.0</td>
      <td>72.0</td>
      <td>10</td>
      <td>162.0</td>
      <td>42.0</td>
      <td>20.0</td>
      <td>159</td>
      <td>176.0</td>
      <td>379.0</td>
      <td>184.0</td>
      <td>70.0</td>
      <td>6.0</td>
      <td>16.0</td>
      <td>187.0</td>
      <td>197</td>
      <td>van</td>
    </tr>
    <tr>
      <th>1</th>
      <td>91</td>
      <td>41.0</td>
      <td>84.0</td>
      <td>141.0</td>
      <td>57.0</td>
      <td>9</td>
      <td>149.0</td>
      <td>45.0</td>
      <td>19.0</td>
      <td>143</td>
      <td>170.0</td>
      <td>330.0</td>
      <td>158.0</td>
      <td>72.0</td>
      <td>9.0</td>
      <td>14.0</td>
      <td>189.0</td>
      <td>199</td>
      <td>van</td>
    </tr>
    <tr>
      <th>2</th>
      <td>104</td>
      <td>50.0</td>
      <td>106.0</td>
      <td>209.0</td>
      <td>66.0</td>
      <td>10</td>
      <td>207.0</td>
      <td>32.0</td>
      <td>23.0</td>
      <td>158</td>
      <td>223.0</td>
      <td>635.0</td>
      <td>220.0</td>
      <td>73.0</td>
      <td>14.0</td>
      <td>9.0</td>
      <td>188.0</td>
      <td>196</td>
      <td>car</td>
    </tr>
    <tr>
      <th>3</th>
      <td>93</td>
      <td>41.0</td>
      <td>82.0</td>
      <td>159.0</td>
      <td>63.0</td>
      <td>9</td>
      <td>144.0</td>
      <td>46.0</td>
      <td>19.0</td>
      <td>143</td>
      <td>160.0</td>
      <td>309.0</td>
      <td>127.0</td>
      <td>63.0</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>199.0</td>
      <td>207</td>
      <td>van</td>
    </tr>
    <tr>
      <th>4</th>
      <td>85</td>
      <td>44.0</td>
      <td>70.0</td>
      <td>205.0</td>
      <td>103.0</td>
      <td>52</td>
      <td>149.0</td>
      <td>45.0</td>
      <td>19.0</td>
      <td>144</td>
      <td>241.0</td>
      <td>325.0</td>
      <td>188.0</td>
      <td>127.0</td>
      <td>9.0</td>
      <td>11.0</td>
      <td>180.0</td>
      <td>183</td>
      <td>bus</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>841</th>
      <td>93</td>
      <td>39.0</td>
      <td>87.0</td>
      <td>183.0</td>
      <td>64.0</td>
      <td>8</td>
      <td>169.0</td>
      <td>40.0</td>
      <td>20.0</td>
      <td>134</td>
      <td>200.0</td>
      <td>422.0</td>
      <td>149.0</td>
      <td>72.0</td>
      <td>7.0</td>
      <td>25.0</td>
      <td>188.0</td>
      <td>195</td>
      <td>car</td>
    </tr>
    <tr>
      <th>842</th>
      <td>89</td>
      <td>46.0</td>
      <td>84.0</td>
      <td>163.0</td>
      <td>66.0</td>
      <td>11</td>
      <td>159.0</td>
      <td>43.0</td>
      <td>20.0</td>
      <td>159</td>
      <td>173.0</td>
      <td>368.0</td>
      <td>176.0</td>
      <td>72.0</td>
      <td>1.0</td>
      <td>20.0</td>
      <td>186.0</td>
      <td>197</td>
      <td>van</td>
    </tr>
    <tr>
      <th>843</th>
      <td>106</td>
      <td>54.0</td>
      <td>101.0</td>
      <td>222.0</td>
      <td>67.0</td>
      <td>12</td>
      <td>222.0</td>
      <td>30.0</td>
      <td>25.0</td>
      <td>173</td>
      <td>228.0</td>
      <td>721.0</td>
      <td>200.0</td>
      <td>70.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>187.0</td>
      <td>201</td>
      <td>car</td>
    </tr>
    <tr>
      <th>844</th>
      <td>86</td>
      <td>36.0</td>
      <td>78.0</td>
      <td>146.0</td>
      <td>58.0</td>
      <td>7</td>
      <td>135.0</td>
      <td>50.0</td>
      <td>18.0</td>
      <td>124</td>
      <td>155.0</td>
      <td>270.0</td>
      <td>148.0</td>
      <td>66.0</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>190.0</td>
      <td>195</td>
      <td>car</td>
    </tr>
    <tr>
      <th>845</th>
      <td>85</td>
      <td>36.0</td>
      <td>66.0</td>
      <td>123.0</td>
      <td>55.0</td>
      <td>5</td>
      <td>120.0</td>
      <td>56.0</td>
      <td>17.0</td>
      <td>128</td>
      <td>140.0</td>
      <td>212.0</td>
      <td>131.0</td>
      <td>73.0</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>186.0</td>
      <td>190</td>
      <td>van</td>
    </tr>
  </tbody>
</table>
<p>846 rows × 19 columns</p>
</div>




```python
label_enc = preprocessing.LabelEncoder()
# The target variable is "class"
label_enc.fit(df['class'])
df['class']=label_enc.fit_transform(df['class'])

X = df.drop('class', 1)
Y = df['class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 50)

```

## Part B


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pipe = Pipeline(steps=[
    ('simple imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
('scaler', StandardScaler()),('DTClassifier', DecisionTreeClassifier())
])
pipe.fit(X_train, Y_train)
print(classification_report(Y_test, pipe.predict(X_test)))
print(pipe.score(X_test, Y_test))
```

                  precision    recall  f1-score   support
    
               0       0.93      0.93      0.93        40
               1       0.95      0.90      0.93        91
               2       0.84      0.95      0.89        39
    
        accuracy                           0.92       170
       macro avg       0.91      0.92      0.91       170
    weighted avg       0.92      0.92      0.92       170
    
    0.9176470588235294
    

## Part C


```python
from sklearn.model_selection import GridSearchCV
Newpipe = Pipeline(steps=[
    ('simple imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
('scaler', StandardScaler()),('pca', PCA()),('DTClassifier', DecisionTreeClassifier())])

param_grid = {
        'DTClassifier__max_depth': [3, 5, 15],
        'pca__n_components': [5, 8, 10],
}


grid_search = GridSearchCV(Newpipe, param_grid=param_grid)

grid_search.fit(X_train, Y_train)
print('The best paramaters for Decision tree and PCA are', grid_search.best_params_)
```

    The best paramaters for Decision tree and PCA are {'DTClassifier__max_depth': 15, 'pca__n_components': 10}
    


```python
Newpipe.set_params(DTClassifier__max_depth=15, pca__n_components=10)
Newpipe.fit(X_train, Y_train)
print(classification_report(Y_test, Newpipe.predict(X_test)))
print(Newpipe.score(X_test, Y_test))
```

                  precision    recall  f1-score   support
    
               0       0.81      0.72      0.76        40
               1       0.88      0.89      0.89        91
               2       0.74      0.79      0.77        39
    
        accuracy                           0.83       170
       macro avg       0.81      0.80      0.80       170
    weighted avg       0.83      0.83      0.83       170
    
    0.8294117647058824
    

# Question 4 (15pts) - Reject option

Consider a binary classification problem with the following loss matrix - where the cost of rejection is a constant. 

$$
   {\begin{array}{ccccc}
   & & \text{Predicted class} & \text{           } &\\
   & & C1 & C2 & Reject\\
   \text{True class} & C1 & 0 & 3 & c  \\
   & C2 & 2 & 0 & c \\
  \end{array} } 
$$

Determine the prediction that minimizes the expected loss in different ranges of $P(C1|x)$ where c = 1

Loss of choosing $C_1$:
$$
0P(C_1|x) + 2(1-P(C_1|x)) = 2-2P(C_1|x)
$$
Loss of choosing $C_2$:
$$
3P(C_1|x) + 0(1-P(C_1|x)) = 3P(C_1|x)
$$
Loss of choosing to reject:
$$
cP(C_1|x)+c(1-P(C_1|x)) = c = 1
$$


Decision boundary of $C_1$ and Reject:
$$
\begin{eqnarray*}
2-2P(C_1|x) &=& 1 \\
P(C_1|x) &=& \frac{1}{2}
\end{eqnarray*}
$$
Decision boundary of $C_2$ and Reject:
$$
\begin{eqnarray*}
3P(C_1|x) &=& 1 \\
P(C_1|x) &=& \frac{1}{3}
\end{eqnarray*}
$$
Decision boundary of $C_1$ and $C_2$:
$$
\begin{eqnarray*}
2-2P(C_1|x) &=& 3P(C_1|x) \\
P(C_1|x) &=& \frac{2}{5}
\end{eqnarray*}
$$

Notably the decision boundaries have the following relationship $\frac{1}{3} < \frac{2}{5} < \frac{1}{2}$. This makes it easier to visualize the decision boundaries and where to choose certain classes.  
Since the loss function for choosing class 2 is lower at the beginning, we **choose class 2 for $P(C_1|x) < \frac{1}{3}$**  
  
We **choose to reject in the range $\frac{1}{3}<\frac{1}{2}$** as the rejection loss is a constant 1 while the other two losses are higher in that range. 
  
For **$P(C_1|x) > \frac{1}{2}$ we choose class 1** as that minimizes loss.  
  
Lastly, we are indifferent between rejecting and choosing $C_2$ for $\frac{1}{3}$ and we are indifferent between choosing $C_1$ and rejecting for $\frac{1}{2}$  
  
We can confirm this analysis with the following plot:


```python
#use np.linspace to create an array to graph various probability values
prob = np.linspace(0,1,100)

#Define loss functions
c1_loss = 2-2*prob
c2_loss = 3*prob
reject_loss = np.ones(len(prob))

#Make a scatterplot
plt.scatter(prob,c1_loss,color = 'blue', label = 'Loss of Choosing Class 1')
plt.scatter(prob,c2_loss,color = 'red', label = 'Loss of Choosing Class 2')
plt.scatter(prob,reject_loss,color = 'grey', label ='Loss of Choosing to Reject')
plt.title('Solution Sketch of Expected Loss')
plt.xlabel('P(C_1|X)')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


    
![png](output_57_0.png)
    


# Question 5: Supplementary Notes (10 points)
Visit the [Supplementary Notes website](https://ideal-ut.github.io/APM-2020/). Read the notes written by your peers from both sections (7th topic onwards, i.e. "stochastic gradient descent" onwards). Select the note that you liked the most and write the following:
- Identify the note. (e.g., If you liked the note from Section A for topic 7, write 7A).
- Write one-two paragraphs on why you liked your selection the most and what you learnt from it.
- Also write a short paragraph on how you think this note can be further improved.

Again for note criteria judging, we are basing it off of the following criteria. 1. Presentation, 2. Time it takes to understand the note, 3. Brevity, 4. Covering material not in lecture, 5. Amount of depth fit in the note. With this criteria, we enjoyed looking at **14B**. The note had excellent visualizations overall and also kept the language clean and concise in such a way that a heavy math background is not required. It also elaborates more on questions present in lecture such as differentiating between PCA and Linear Regression. Overall, it was excellent under our criteria. The only thing that we would add in are some more supplemental resources. 

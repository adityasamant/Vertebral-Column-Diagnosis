import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns; sns.set(style="white", color_codes=True)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import math


# Read Column_2C.dat file
data = pd.read_csv("column_2C.dat",sep="\s+",header=None)

# Give Column Names
data.columns = ['pelvic_incidence','pelvic_tilt',
                'lumbar_lordosis_angle','sacral_slope',
                'pelvic_radius','degree_spondylolisthesis','class']
data.index.name = None 

# Map Class Names to Bool Integers
classmap = {'AB': 1, 'NO': 0}
data = data.replace({'class': classmap})

# Show Data
data.head(10)

# Using seaborn pairgrid to create scatterplot matrix
scatplot = sns.PairGrid(data,hue = "class", hue_kws={"marker" : [ "D","o"]},
                 palette=["#104599","#980002"],
                 vars=['pelvic_incidence','pelvic_tilt',
                       'lumbar_lordosis_angle','sacral_slope',
                       'pelvic_radius','degree_spondylolisthesis'])

# Plotting
scatplot = scatplot.add_legend()
scatplot = scatplot.map(plt.scatter)

# Setting Size for Box Plots and adjusting width between them
plt.figure(figsize=(20,20))
plt.subplots_adjust(wspace=1)

# Creating Each Boxplot Recursively
for axis in range(len(data.columns)-1):
    plt.subplot(2,3,axis+1)
    sns.boxplot(x="class",y=data.columns[axis],palette=["#104599","Red"],data=data)

# Splitting and Extracting Train Data
train0 = data[data['class']==0].head(70)
train1=data[data['class']==1].head(140)
train=pd.concat([train0,train1],axis=0)
y_train = train['class']
X_train = train.drop('class', axis=1)

# Splitting and Extracting Test Data
test0 = data[data['class']==0][70:]
test1=data[data['class']==1][140:]
test=pd.concat([test0,test1],axis=0)
y_test = test['class']
X_test = test.drop('class', axis=1)

# Getting Unified Data
y = data['class']
X = data.drop('class', axis=1)

# Printing Size of Train and Test Data
print("Train Data Size: ",len(train),"\tTest Data Size: ",len(test))


# Creating a model for K nearest neighbours Classifier
knn_default = KNeighborsClassifier()

# Fitting model based on Train data
knn_default.fit(X_train, y_train)

# Predicting classes for Test data using Trained Model
y_pred = knn_default.predict(X_test)

# Printing the model Accuracy
print("The model is ",accuracy_score(y_test, y_pred)*100,"% accurate.")

# Initiating Error Lists
Error_Train = []
Error_Test = []

# Looping through each value of k [k=1 causes overfitting and thus being ignored]
for k in range(210,1,-1):
    knn_bestk = KNeighborsClassifier(n_neighbors=k)
    
    # Storing Train and Test Errors for each k in Lists
    Error_Train.append(1-knn_bestk.fit(X_train, y_train).score(X_train, y_train))
    Error_Test.append(1-knn_bestk.score(X_test, y_test))
    
# Plotting Test and Train Errors against the Value of K neighbours
plt.plot(range(210,1,-1),Error_Test,'r', label="Test Error")
plt.plot(range(210,1,-1),Error_Train,'b', label="Train Error")
plt.xlabel("Value of K")
plt.xlim(210, 1)
plt.ylabel("Error Rate")
plt.legend()
plt.show()

# Printing the Best Value of K using Min Test Error
bestk = 210-Error_Test.index(min(Error_Test))
print("Best value of k is ",bestk)

# Running Instance of Knn Model with best K
knn_cm = KNeighborsClassifier(n_neighbors=bestk)
knn_cm.fit(X_train, y_train)
y_pred_cm = knn_cm.predict(X_test)

# Calculating the Confusion Matrix
confM = confusion_matrix(y_test,y_pred_cm)

# Converting Confusion Matrix to Data Frame and Plotting
DataFrame_confM = pd.DataFrame(confM, index = ["Normal","Abnormal"],columns = ["Normal","Abnormal"])
plt.figure(figsize = (7,7))
axis = sns.heatmap(DataFrame_confM, annot=True, cbar=False, cmap="Reds")

# Decomposing Confusion Matrix
TP = confM[1][1] # True Positive
FP = confM[0][1] # False Positive
FN = confM[1][0] # False Negative
TN = confM[0][0] # True Negative

# True positive rate, Recall
TPR = TP/(TP+FN)
# True negative rate
TNR = TN/(TN+FP) 
# Precision
PPV = TP/(TP+FP)
# F Score
FSc = 2*(PPV*TPR)/(PPV+TPR)

# Printing Values
print("True Positive Rate: \t",round(TPR,3))
print("True Negative Rate: \t",round(TNR,3))
print("Precision: \t\t",round(PPV,3))
print("F Score: \t\t",round(FSc,3))

# List to store min error for each n
Error_N = []
Best_k = []

# Loop each n and k values to find optimal error rate
for n in range(1,210,1):
    
    # Division of DATA in N and N-N/3
    train0_lc = train[train['class']==0].head(math.floor(n/3))
    train1_lc = train[train['class']==1].head(n-math.floor(n/3))
    train_lc = pd.concat([train0_lc,train1_lc],axis=0)
    y_train_lc = train_lc['class']
    X_train_lc = train_lc.drop('class', axis=1)

    
    # Best Error Data for each N
    Error_Test = []
    for k in range(1,n+1,1):
        knn_bestn = KNeighborsClassifier(n_neighbors=k)
        Error_Test.append(1-knn_bestn.fit(X_train_lc, y_train_lc).score(X_test, y_test))
    Error_N.append(min(Error_Test))
    
    # Best k for each N
    Error_Test = []
    for k in range(1,n+1,5):
        knn_bestn = KNeighborsClassifier(n_neighbors=k)
        Error_Test.append(1-knn_bestn.fit(X_train_lc, y_train_lc).score(X_test, y_test))
    Best_k.append(1+5*Error_Test.index(min(Error_Test)))
    
# Plotting Learning Curve
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(range(1,210,1),Error_N,'r') # ax1 = error vs N [RED]
ax1.set_ylabel('Error Rate', color='r')
ax1.set_xlabel('Value of N')
ax1.tick_params(axis='y', labelcolor='r')
ax2 = ax1.twinx()
ax2.plot(range(1,210,1),Best_k, 'b') # ax2 = best k vs N [BLUE]
ax2.set_ylabel('Value of K', color='b')
ax2.tick_params(axis='y', labelcolor='b')
fig.tight_layout()
plt.show()

Error_DM = []


Error_Test_Man = []

# Looping through all value of K
for k in range(1,197,5):
    knn_man = KNeighborsClassifier(n_neighbors=k,p=1) # manhattan = Minkowski with (p=1)
    Error_Test_Man.append(1-knn_man.fit(X_train, y_train).score(X_test, y_test))

# Adding test errors of Manhattan Metric to table
Error_DM.append(Error_Test_Man)

# Printing Best Value of K
print("Best value of k is",1+5*Error_Test_Man.index(min(Error_Test_Man)))

Error_Test_Log = []

# Looping through all values of log10(p)
ip = 0.1
while ip<1.01:
    knn_log = KNeighborsClassifier(n_neighbors=6,p = math.pow(10,ip))
    Error_Test_Log.append(1-knn_log.fit(X_train, y_train).score(X_test, y_test))
    ip=round(ip+0.1,1)

# Printing Best Value of log10(p)
Best_log = 0.1+(Error_Test_Log.index(min(Error_Test_Log)))/10
print("The Best log\u2081\u2080(p) is",Best_log)

# Code to add test errors of (d)iB to Table
Error_Test_LogAdd = []

# Looping through all value of K
for k in range(1,197,5):
    knn_logadd = KNeighborsClassifier(n_neighbors=k,p = math.pow(10,Best_log))
    Error_Test_LogAdd.append(1-knn_logadd.fit(X_train, y_train).score(X_test, y_test))

# Adding test errors of Minkowski Metric to table
Error_DM.append(Error_Test_LogAdd)

Error_Test_Che = []

# Looping through all value of K
for k in range(1,197,5):
    knn_che = KNeighborsClassifier(n_neighbors=k,metric="chebyshev")
    Error_Test_Che.append(1-knn_che.fit(X_train, y_train).score(X_test, y_test))

# Adding test errors of Chebyshev Metric to table
Error_DM.append(Error_Test_Che)

# Printing Best Value of K
print("Best value of k is",1+5*Error_Test_Che.index(min(Error_Test_Che)))

Error_Test_Mah = []

# Looping through all value of K
for k in range(1,197,5):
    knn_mah = KNeighborsClassifier(n_neighbors=k, metric='mahalanobis', metric_params={'V':X.cov()})
    Error_Test_Mah.append(1-knn_mah.fit(X_train, y_train).score(X_test, y_test))

# Adding test errors of Mahalanobis Metric to table
Error_DM.append(Error_Test_Mah)

# Printing Best Value of K
print("Best value of k is",1+5*Error_Test_Mah.index(min(Error_Test_Mah)))

# Print Best K and Test Error for every Metric
print("DISTANCE METRIC\t\t\t BEST K\t\t BEST TEST ERROR")
print("---------------------------------------------------------------------")
print("Manhattan Metric\t\t",1+5*Error_Test_Man.index(min(Error_Test_Man)),"\t\t",min(Error_Test_Man))
print("Minkowski log₁₀(p) = 0.6\t",1+5*Error_Test_LogAdd.index(min(Error_Test_LogAdd)),"\t\t",min(Error_Test_LogAdd))
print("Chebyshev Metric\t\t",1+5*Error_Test_Che.index(min(Error_Test_Che)),"\t\t",min(Error_Test_Che))
print("Mahalanobis Metric\t\t",1+5*Error_Test_Mah.index(min(Error_Test_Mah)),"\t\t",min(Error_Test_Mah))

# Adding Values of K to Matrix for Reference
k_iterator= []
for i in range(1,197,5):
    k_iterator.append(i)
Error_DM.append(k_iterator)

# Transpose of Matrix
Error_DM = list(map(list, zip(*Error_DM)))

# Converting Matrix to DataFrame
comparisonM = pd.DataFrame(Error_DM)
comparisonM.columns = ['Manhattan',"Minkowski log\u2081\u2080(p) = 0.6",
                 'Chebyshev','Mahalanobis','Value of K']

# Displaying Table with top 15 comparisons
comparisonM.head(15)
Error_Weighted = []
Error_Test_EucW = []
Error_Test_ManW = []
Error_Test_CheW = []
Error_Test_MinW = []
kval = []

# Looping through all value of K for Euclidean, Manhattan and Chebyshev Metrics
for k in range(1,197,5):
    # Euclidean Metric
    knn_eucw = KNeighborsClassifier(n_neighbors=k, p = 2, weights="distance")
    Error_Test_EucW.append(1-knn_eucw.fit(X_train, y_train).score(X_test, y_test))
    
    # Manhattan Metric
    knn_manw = KNeighborsClassifier(n_neighbors=k, p = 1, weights="distance")
    Error_Test_ManW.append(1-knn_manw.fit(X_train, y_train).score(X_test, y_test))
    
    # Chebyshev Metric
    knn_chew = KNeighborsClassifier(n_neighbors=k, metric="chebyshev", weights="distance")
    Error_Test_CheW.append(1-knn_chew.fit(X_train, y_train).score(X_test, y_test))
    
    Error_Test_MinW.append(min(Error_Test_EucW[round(k/5)],Error_Test_ManW[round(k/5)],Error_Test_CheW[round(k/5)]))
    kval.append(k)
    
# Print Best K and Test Error for every Metric
print("DISTANCE METRIC\t\t BEST TEST ERROR")
print("--------------------------------------------")
print("Euclidean Metric\t",min(Error_Test_EucW))
print("Manhattan Metric\t",min(Error_Test_ManW))
print("Chebyshev Metric\t",min(Error_Test_CheW))

# Adding test errors of Weighted Metrics to table
Error_Weighted.append(Error_Test_EucW) # Euclidean Metric
Error_Weighted.append(Error_Test_ManW) # Manhattan Metric
Error_Weighted.append(Error_Test_CheW) # Chebyshev Metric
Error_Weighted.append(Error_Test_MinW)
Error_Weighted.append(kval)

# Transpose
Error_Weighted = list(map(list, zip(*Error_Weighted)))

# Converting Matrix to DataFrame
compW=pd.DataFrame(Error_Weighted)
compW.columns = ['Euclidean','Manhattan','Chebyshev','Minimun Test Error','Value of K']

# Displaying Table with top 15 comparisons
compW.head(15)

knn_lte = KNeighborsClassifier(weights="distance")
print("Training error is",1-knn_lte.fit(X_train, y_train).score(X_train, y_train))


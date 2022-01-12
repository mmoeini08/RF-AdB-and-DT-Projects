import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from dtreeviz.trees import dtreeviz
sns.set(style='darkgrid')
file_name = 'TSS'
data = pd.read_csv(file_name + '.csv', header=0, index_col=0)
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")
X = data.drop(columns=['TSS'])
Y = data.TSS
data1=[]
data2=[]
data3=[]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
cross_val_number = 5
decitree = tree.DecisionTreeRegressor(max_depth=3)
decitree_scores = cross_val_score(decitree, X_train_scaled, Y_train, cv=cross_val_number)
decitree.fit(X_train_scaled, Y_train)
print("")
print("Decision Tree R-Squared: {0} (+/- {1})".format(decitree_scores.mean().round(2), (decitree_scores.std() * 2).round(2)))
decitree_predict = decitree.predict(X_test_scaled)
decitree_score = metrics.r2_score(decitree_predict, Y_test)
print("Decision Tree: {0}".format(decitree_predict))
print("Decision tree score: {0}".format(decitree_score))
print("")

for i in range(1,101):
    data3.append(i)
    randfor = RandomForestRegressor(n_estimators=i, max_samples=20, max_depth=3)
    gbm = GradientBoostingRegressor(n_estimators=i, max_depth=3)
    randfor.fit(X_train_scaled, Y_train) #Random Forest fitting
    gbm.fit(X_train_scaled, Y_train)
    randfor_scores = cross_val_score(randfor, X_train_scaled, Y_train, cv=cross_val_number)
    data1.append(randfor_scores.mean().round(2))
    gbm_scores = cross_val_score(gbm, X_train_scaled, Y_train, cv=cross_val_number)
    data2.append(gbm_scores.mean().round(2))
    randfor_predict = randfor.predict(X_test_scaled)
    randfor_score = metrics.r2_score(randfor_predict, Y_test)
    gbm_predict = gbm.predict(X_test_scaled)
    gbm_score = metrics.r2_score(gbm_predict, Y_test)
fig = plt.figure(figsize=(12, 4))
fig.tight_layout(pad=5.0)
y1=data1
y2=data2 
x = np.linspace(1, 101, 100)
ax = fig.add_subplot(121)
ax.set_title("Random Forest regression", fontsize=20)
ax.plot(x,y1)
ax.set_xlabel('Number of estimators', fontsize=15)
ax.set_ylabel('R-squared', fontsize=15)
ax = fig.add_subplot(122)
ax.set_title("Gradient Boosting regression", fontsize=20)
ax.set_xlabel('Number of estimators', fontsize=15)
ax.set_ylabel('R-squared', fontsize=15)
ax.plot(x,y2)
plt.savefig('RF_GB_Regre_TSS).png', dpi=300)
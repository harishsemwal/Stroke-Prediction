import pandas as pd
import matplotlib.pyplot as plt
#from sympy.physics.control.control_plots import matplotlib
#%matplotlib inline
import seaborn as sns
data=pd.read_csv('healthcare-dataset-stroke-data.csv')
data.info()
data.isnull().sum()
data['bmi'].describe()
data['bmi'].fillna(data['bmi'].mean(),inplace=True)
data['bmi'].describe()
# feature selection-dropping id coloumn because it is of no use here
data.drop('id',axis=1,inplace=True)
# outlier detection
plt.rcParams['figure.figsize']=(10,6)
data.plot(kind='box')
plt.show()
data['avg_glucose_level'].describe()
print(data[data['avg_glucose_level']>114.090000])
# Label Encoding
data['work_type'].unique()
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
data['gender']=enc.fit_transform(data['gender'])
data['ever_married']=enc.fit_transform(data['ever_married'])
data['work_type']=enc.fit_transform(data['work_type'])
data['Residence_type']=enc.fit_transform(data['Residence_type'])
data['smoking_status']=enc.fit_transform(data['smoking_status'])
data['work_type'].unique()
print(data)
#data partitioning = splitting the data for train and test
#x-->x_train,x_test     y-->y_train,y_test (80/20)
x=data.drop('stroke',axis=1)
y=data['stroke']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)
# Normalisation of input data features
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
x_train_std=std.fit_transform(x_train)
x_test_std=std.transform(x_test)
import pickle
import os
scaler_path=os.path.join('C:/Users/lenovo/Machine Learning Projects/stroke prediction/','models/scaler.pkl')
with open(scaler_path,'wb')as scaler_file:
    pickle.dump(std,scaler_file)

print(x_train_std)
print(x_test_std)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train_std,y_train)
y_pred=dt.predict(x_test_std)
from sklearn.metrics import accuracy_score
ac_dt=accuracy_score(y_test,y_pred)
print(ac_dt)
import joblib
model_path=os.path.join('C:/Users/lenovo/Machine Learning Projects/stroke prediction/','models/dt.sav')
joblib.dump(dt,model_path)
#by using hyperparameter tuning we can improve the accuracy but here we are not doing any hyperparameter tuning
#logistic regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train_std,y_train)
y_pred=lr.predict(x_test_std)
ac_lr=accuracy_score(y_test,y_pred)
print(ac_lr)
#k nearest neighbors(by default it takes 'k' value  5 here)
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
knn.fit(x_train_std,y_train)
y_pred=knn.predict(x_test_std)
ac_knn=accuracy_score(y_test,y_pred)
print(ac_knn)
#random forest=collection of so many decision trees and selecting the majority answer among there results as answer
#random forest and decision tree algo do not require stardized data(x_train_std,x_test_std) as they are not calculating any distance while giving results.
#random forest and decision trees can also use normal data(x_train,x_test) to claculate accuracy.it do not neccessarily need standardized data(x_train_std,x_test_std).
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train_std,y_train)
y_pred=rf.predict(x_test_std)
ac_rf=accuracy_score(y_test,y_pred)
print(ac_rf)
#svm
from sklearn.svm import SVC
sv=SVC(kernel='sigmoid',gamma=1.0)
sv.fit(x_train_std,y_train)
y_pred=sv.predict(x_test_std)
ac_sv=accuracy_score(y_test,y_pred)
print(ac_sv)
#voting classifier(to use multiple algorithms together(ensembling of diff. algos) to check change in accuracy and precision)
lr=LogisticRegression()
knn= KNeighborsClassifier()
rf=RandomForestClassifier()
from sklearn.ensemble import VotingClassifier
voting=VotingClassifier(estimators=[('lr',lr),('knn',knn),('rfc',rf)],voting='soft')
voting.fit(x_train_std,y_train)
y_pred=voting.predict(x_test_std)
ac_voting=accuracy_score(y_test,y_pred)
print("Accuracy",accuracy_score(y_test,y_pred))
# Applying stacking(a method for ensembling different algos to check change in accuracy)
estimators=[('lr',lr),('knn',knn),('rfc',rf)]
final_estimator=LogisticRegression()
from sklearn.ensemble import StackingClassifier
clf=StackingClassifier(estimators=estimators,final_estimator=final_estimator)
clf.fit(x_train_std,y_train)
y_pred=clf.predict(x_test_std)
ac_stacking=accuracy_score(y_test,y_pred)
print("Accuracy",accuracy_score(y_test,y_pred))
plt.bar(['decision trees','logistic reg','knn','random forest','svm','voting','stacking'],[ac_dt,ac_lr,ac_knn,ac_rf,ac_sv,ac_voting,ac_stacking])
plt.xlabel("<----Algorithms---->")
plt.ylabel("<----Accuracy---->")
plt.show()
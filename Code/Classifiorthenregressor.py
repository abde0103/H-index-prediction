from sklearn.svm import SVC
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
Train=pd.read_csv(r"C:\Users\moham\Desktop\info\99_embedcsv-train.csv", index_col=0)
Test=pd.read_csv(r"C:\Users\moham\Desktop\info\99_embedcsv-test.csv", index_col=0)
Train = Train.loc[:, ~Train.columns.str.contains('^Unnamed')]
Test = Test.loc[:, ~Test.columns.str.contains('^Unnamed')]
Y_Train=Train.loc[:,"hindex"].values 
del Train ["hindex"]
del Train["node"]
X_Train=Train.values
Y_Test=Test.loc[:,"hindex"].values
del Test ["hindex"]
del Test["node"]
X_Test=Test.values 
Max=X_Train.max(axis=1)
Min=X_Train.min(axis=1)
Max=Max.reshape((Max.shape[0],1))
Min=Min.reshape((Min.shape[0],1))
Max=np.tile(Max,X_Train.shape[1])
Min=np.tile(Min,X_Train.shape[1])
X_Train=(-X_Train+Max)/(Max-Min)
Max=X_Test.max(axis=1)
Min=X_Test.min(axis=1)
Max=Max.reshape((Max.shape[0],1))
Min=Min.reshape((Min.shape[0],1))
Max=np.tile(Max,X_Test.shape[1])
Min=np.tile(Min,X_Test.shape[1])
X_Test=(-X_Test+Max)/(Max-Min)
Y_Train_classifier=np.copy(Y_Train)
for i,j in enumerate(Y_Train):
       if  Y_Train[i]<85 :
              Y_Train_classifier[i]=0
       elif 85<Y_Train[i]<105 :
              Y_Train_classifier[i]=1
       else : 
         Y_Train_classifier[i]=2
print(Y_Test)
Y_Test_classifier=np.copy(Y_Test)
for i,j in enumerate(Y_Test):
       if  Y_Test[i]<84 :
              Y_Test_classifier[i]=0
       elif 84<Y_Test[i]<105 :
              Y_Test_classifier[i]=1
       else : 
         Y_Test_classifier[i]=2
print(Y_Test_classifier)
print(Y_Test)

classifier = KNeighborsClassifier(n_neighbors=10,weights="distance")
classifier.fit(X_Train, Y_Train_classifier)
y_predclass = classifier.predict(X_Test)
print(classification_report(Y_Test_classifier,y_predclass))









XTrain_v2=[]
YTrain_v2=[]
for i,j in enumerate(Y_Train):
    if 105>Y_Train[i]>84:
        XTrain_v2.append(X_Train[i])
        YTrain_v2.append(Y_Train[i])
        
        
XTrain_v3=[]
YTrain_v3=[]
for i,j in enumerate(Y_Train):
    if Y_Train[i]>=105:
        XTrain_v3.append(X_Train[i])
        YTrain_v3.append(Y_Train[i])



modelv2= KNeighborsRegressor(n_neighbors=10,weights="distance")
modelv3 = KNeighborsRegressor(n_neighbors=10,weights="distance")
modelv2.fit(XTrain_v2,YTrain_v2)
modelv3.fit(XTrain_v3,YTrain_v3)








from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
modelv1 = XGBRegressor(n_estimators=10, max_depth=10, eta=0.1, subsample=1, colsample_bytree=1, eval_metric=mean_squared_error)
modelv1.fit(X_Train, Y_Train,eval_set  =[(X_Test, Y_Test)])



X_v1=X_Test[y_predclass==0]
Y_v1=Y_Test[y_predclass==0]
X_v2=X_Test[y_predclass==1]
Y_v2=Y_Test[y_predclass==1]
X_v3=X_Test[y_predclass==2]
Y_v3=Y_Test[y_predclass==2]



Y_Predict_v1=modelv1.predict(X_v1)
Y_Predict_v2=modelv3.predict(X_v2)
Y_Predict_v3=modelv3.predict(X_v3)



error1=(Y_Predict_v1-Y_v1)**2
error2=(Y_Predict_v2-Y_v2)**2
error3=(Y_Predict_v3-Y_v3)**2
print(error1/error1.shape[0])
print(error2/error2.shape[0])
print(error3/error3.shape[0])
print((sum(error1)+sum(error2)+sum(error3))/(error1.shape[0]+error2.shape[0]+error3.shape[0]))

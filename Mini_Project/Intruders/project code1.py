import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
Dataset=pd.read_csv('files.csv')
print(Dataset.describe())

x=Dataset.iloc[:,:-1].values
x1=pd.DataFrame(x)
y=Dataset.iloc[:,10].values
y1=pd.DataFrame(y)

for i in range(119):
    if y[i]=='anom':
        y[i]=0
    else:
        y[i]=1
type(y)
type(x)
y=y.astype('int')

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
for i in range(10) :
    x[:,i]=labelencoder_x.fit_transform(x[:,i])
    Y=pd.DataFrame(x[:,i])
for i in range(10) :
    #onehotencoder=OneHotEncoder(categorical_features=[i])
    #x=onehotencoder.fit_transform(x).toarray()
    onehotencoder = OneHotEncoder()
    X = onehotencoder.fit_transform(x).toarray()
    X = X[:, 1:]


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#write in file
np.savetxt('encode_valuse.txt',x)

#Missing Data Removal
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values='NaN', strategy='most_frequent')
imputer = imputer.fit(x[:,:])
x[:,:]=imputer.transform(x[:,:])
Missing_Data_Removed=imputer.transform(x[:,:])

#write in file
np.savetxt('Missing_values.txt',Missing_Data_Removed)

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0) 
 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#SVM Apply
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)

#prediction
y_pred = svclassifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

dt = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
#performance anaylsis
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
from sklearn import metrics 
print("Accuracy:",accuracy_score(y_test,y_pred)*100)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
Total_TP_FP=cm[0][0]+cm[0][1]
Total_FN_TN=cm[1][0]+cm[1][1]

#True Positive Calculation
TP1=(((cm[0][0])/Total_TP_FP)*100)

#False Positive Calculation
FP1= (((cm[0][1])/Total_TP_FP)*100)

#False Negative Calculation
FN1=(((cm[1][0])/Total_FN_TN)*100)

#True Negative Calculation
TN1=(((cm[1][1])/Total_FN_TN)*100)

#Total TP,TN,FP,FN
Total=TP1+FP1+FN1+TN1

#Accuracy Calculation
accuracy=((TP1+TN1)/Total)*100

#Error Rate Calculation
error_rate=((FP1+FN1)/Total)*100

#Precision Calculation
precision=TP1/(TP1+FP1)*100

#Recall Calculation
recall=TP1/(TP1+FN1)*100

#F1 Score
f1=2*((precision*recall)/(precision+recall))

print("\n\n\tResult Generation")
print("\t------------------")
print('\n\tTrue positive = ', TP1, '%')
print('\n\tFalse positive = ', FP1, '%')
print('\n\tFalse negative = ', FN1, '%')
print('\n\tTrue negative = ',TN1 , '%')
print('\n\tAccuracy = ',accuracy , '%')
print('\n\tError Rate = ',error_rate , '%')
print('\n\tPrecision = ',precision , '%')
print('\n\tRecall = ',recall , '%')
print('\n\tF1-Score = ',f1 , '%')


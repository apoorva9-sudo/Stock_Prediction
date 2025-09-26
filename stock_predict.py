from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
%matplotlib inline

# for data manipulation
import pandas as pd
import numpy as np

# To plot the graphs
import matplotlib.pyplot as plt

#To ignore the wrnings
import warnings
warnings.filterwarnings("ignore")


#Now raeding the stock data of reliance company
df=pd.read_csv('/content/sample_data/RELIANCE.csv')


#Data preprocessing

df.index=pd.to_datetime(df['Date'])
df=df.drop(['Date'],axis='columns')

# now creating the predicting  variables  Open and close are the prices respectively  at certain times

df['Open-Close']=df.Open-df.Close
df['High-Low']=df.High-df.Low


# storing the predictor in variable X as it is the set for training
X=df[['Open-Close','High-Low']]
X.head()

#target  is variable y ( i.e., if  tomorrow's closing price is >  todays closing price y=1 orelse =0)
y=np.where(df['Close'].shift(-1)>df['Close'],1,0)

#now splitting the data into the training and test sets traing to 80% and testing to 20% of data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)

cls=SVC().fit(X_train,y_train)


#calculating the traing accuracy
train_accuracy=accuracy_score(y_train,cls.predict(X_train))
# now test accuracy says
test_accuracy=accuracy_score(y_test,cls.predict(X_test))
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

#The model can actually give the accuracy  by using the different kernel like(linear,polynomial,RBF)


# Linear kernel
cls_linear = SVC(kernel='linear').fit(X_train, y_train)
y_pred_linear = cls_linear.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print(f"Accuracy with Linear Kernel: {accuracy_linear}")

# Polynomial kernel
cls_poly = SVC(kernel='poly', degree=3).fit(X_train, y_train)
y_pred_poly = cls_poly.predict(X_test)
accuracy_poly = accuracy_score(y_test, y_pred_poly)
print(f"Accuracy with Polynomial Kernel (degree=3): {accuracy_poly}")

# RBF kernel (default)
cls_rbf = SVC(kernel='rbf').fit(X_train, y_train)
y_pred_rbf = cls_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"Accuracy with RBF Kernel: {accuracy_rbf}")

# Sigmoid kernel
cls_sigmoid = SVC(kernel='sigmoid').fit(X_train, y_train)
y_pred_sigmoid = cls_sigmoid.predict(X_test)
accuracy_sigmoid = accuracy_score(y_test, y_pred_sigmoid)
print(f"Accuracy with Sigmoid Kernel: {accuracy_sigmoid}")


#now, the built startergy will be implemented  using cls.predict
df['Predicted_sig']=cls.predict(X)
# Calculate daily returns
df['Return'] = df.Close.pct_change()
# Calculate strategy returns
df['Strategy_Return'] = df.Return * df.Predicted_sig.shift(1)
# Calculate Cumulutive returns
df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
df['Cum_Ret'] = df['Return'].cumsum()
plt.figure(figsize=(10,5))
plt.plot(df['Cum_Ret'], color='red', label="Market Return")
plt.plot(df['Cum_Strategy'], color='blue', label="Strategy Return")
plt.legend()
plt.show()




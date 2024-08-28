# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:

To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm : 

1. Import the required libaries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement the training set and the test set of the dataframe.
4. Plot the required graph for both the training data and the test data.
5. Find the values of MSE,MAE and RMSE.

## Program & Output:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ALIYA SHEEMA
RegisterNumber: 212223230011 
*/
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
```
```
df=pd.read_csv("student_scores.csv")
print(df.head())
print(df.tail())
```
![image](https://github.com/user-attachments/assets/262dd93a-52d9-4561-878f-6565abb16bf2)

```
df.info()
```
![image](https://github.com/user-attachments/assets/aad136d8-8afb-43c8-a979-e1b3ac1ed042)

```
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)
```
![image](https://github.com/user-attachments/assets/3c10d80a-b6c1-4db9-97b0-a585a1cbd279)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
```
```
x_train.shape
```
![image](https://github.com/user-attachments/assets/b4c51572-5e54-4a55-bafe-7b134bca3f2d)

```
x_test.shape
```
![image](https://github.com/user-attachments/assets/76427af7-e111-4e3c-b1b2-e0dffabf1999)

```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
```
![image](https://github.com/user-attachments/assets/33b7f2e8-9701-4e89-b74e-65f0a0d3b4e2)
```
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
```
![image](https://github.com/user-attachments/assets/ec347ed7-5133-4cd5-a308-beeb901f5a3b)
```
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,reg.predict(x_train),color="blue")
plt.title("Training set(H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
```
![image](https://github.com/user-attachments/assets/d979ead7-018b-4375-bc00-270a695a40de)

```
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,reg.predict(x_test),color="black")
plt.title("Test set(H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
![image](https://github.com/user-attachments/assets/3981ddf0-82e8-457b-8113-ce8093392b20)

```
mse=mean_squared_error(y_test,y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
![image](https://github.com/user-attachments/assets/8d6d47e9-a727-4d0d-bfe6-6127871e031c)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

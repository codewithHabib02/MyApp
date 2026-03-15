import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings 
warnings.filterwarnings('ignore')
import joblib
lr=pd.read_csv(r'C:\Users\Habibulie\Desktop\project_root\Heart\heart.csv')
print(lr.head())
print(lr.isna().sum())
print(lr.describe())
print(lr.dtypes)
print(lr.info())



plt.figure()
plt.hist(lr['target'],bins=20)
plt.show()

sns.boxplot(data=lr,x='sex',y='target')
plt.show()

select=lr.select_dtypes(include=['int64','float64']).columns 
range=lr[select].min() -lr[select].max()
print(range)

matrix=lr.corr()['target']
print(matrix)

x=lr.drop('target',axis=1)
y=lr[['target']]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=42)
print(np.unique(y_train,return_counts=True))

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

most_freg=y_train.mean()
baseline_acc=np.full(len(y_test),most_freg)

print("r2_score",r2_score(y_test,baseline_acc))
print("MAE",mean_absolute_error(y_test,baseline_acc))
print("MSE",mean_squared_error(y_test,baseline_acc))

model=LinearRegression()
cross_val=cross_val_score(model,x,y,cv=5, scoring='r2')
print(f" cv for each fold:{cross_val}")
print(f"cv mean:{np.mean(cross_val)}")

y_pred_cv=cross_val_predict(model,x,y,cv=5)
print(y_pred_cv)

model.fit(x_train_scaled,y_train)





y_train_predict=model.predict(x_train)
y_test_predidct=model.predict(x_test)
r2_train=r2_score(y_train,y_train_predict)
r2_test=r2_score(y_test,y_test_predidct)
print(f"Train:{r2_train}")
print(f"Test:{r2_test}")

y_pred=model.predict(x_test_scaled)

print(y_pred[:4])
print(y_test[:4])

print("r2_score",r2_score(y_test,y_pred))
print("MAE",mean_absolute_error(y_test,y_pred))
print("MSE",mean_squared_error(y_test,y_pred))

sns.scatterplot(x=np.ravel(y_test),y=np.ravel(y_pred))
plt.plot([np.ravel(y_test).min(),np.ravel(y_test).max()],[np.ravel(y_test).min(),np.ravel(y_test).max()],'r--')
plt.show()


model=LinearRegression()
model.fit(x_train,y_train)
imp=model.coef_[0]
result=pd.DataFrame({
    'feature': x.columns,
    'importance': np.abs(imp)
})
print(result)

sns.barplot(data=result,x='feature',y='importance')
plt.show()
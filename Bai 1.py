import pandas as pd
import matplotlib.pyplot as plt
# Cau 1 Đọc file diabetes.csv vào biến df_diabetes. Hiển thị 10 dòng đầu.
df_diabetes = pd.read_csv("diabetes.csv") # doc file csv
print(df_diabetes.head(10))
# Cau 2 Gán y bằng cột Outcome và X là các cột còn lại của df_diabetes. Hiển thị kích cỡ của X, y
X = df_diabetes.drop(["Outcome"], axis = 1) # X bo cot Outcome
y = df_diabetes["Outcome"] # y chi lay cot Outcome
print(X)
print(y)
# Cau 3 Hiển thị df_diabetes, X, y ở dòng thứ 5
print(df_diabetes.iloc[5])
print('X:',X.iloc[5])
print('y:',y.iloc[5])
# Cau 4 Tính max, min, mean, var và std của cột Age
maxAge=df_diabetes['Age'].max()
minAge=df_diabetes['Age'].min()
meanAge=df_diabetes['Age'].mean()
varAge=df_diabetes['Age'].var()
stdAge=df_diabetes['Age'].std()
print('Max:',maxAge)
print('Min:',minAge)
print('Mean:',meanAge)
print('Var:',varAge)
print('Std:',stdAge)
# Cau 5 Chuẩn hoá min-max cho cột Isulin của X
df_insu = X.copy()
df_insu['Insulin'] = (df_insu['Insulin'] - df_insu['Insulin'].min()) /( df_insu['Insulin'].max() - df_insu['Insulin'].min())
maxInsulin = df_insu['Insulin'].max()
minInsulin = df_insu['Insulin'].min()
print('Max Insulin : ', maxInsulin)
print('Min Insulin : ', minInsulin)
# Cau 6 Gán X2 là 2 cột BMI và Age. Vẽ biểu đồ phân bố 2 lớp 0/1 với X2 và y
X2=df_diabetes[['BMI','Age']]
X0=X2[y==0]
X1=X2[y==1]
plt.plot(X0['BMI'], X0['Age'], 'b^', markersize=4, alpha=.8)
plt.plot(X1['BMI'], X1['Age'], 'go', markersize=4, alpha=.8)
plt.xlabel('BMI')
plt.ylabel('Age')
plt.title('Nhãn lớp 0 và lớp 1')
plt.plot()
plt.show()
# Cau 7 Đếm số phần tử 0 và 1 của y.
print('Tong so phan tu 0 la:',y[y==0].shape[0])
print('Tong so phan tu 1 la:',y[y==1].shape[0])
# Cau 9 Chia tập (X, y) thành (X_train, y_train) và (X_test, y_test) theo tỉ lệ 70/30. Hiển thị kích cỡ
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=35)
print("X_train=")
print(X_train)
print("y_train=")
print(y_train)
print("X_test=")
print(X_test)
print("y_test=")
print(y_test)
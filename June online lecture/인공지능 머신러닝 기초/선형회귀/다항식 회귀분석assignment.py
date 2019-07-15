import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import csv


'''
./data/Advertising.csv ���� �����͸� �о�, X�� Y�� ����ϴ�.

X�� (200, 3) �� shape�� ���� 2���� np.array,
Y�� (200,) �� shape�� ���� 1���� np.array���� �մϴ�.

X�� FB, TV, Newspaper column �� �ش��ϴ� �����͸� �����ؾ� �մϴ�.
Y�� Sales column �� �ش��ϴ� �����͸� �����ؾ� �մϴ�.
'''
csvreader = csv.reader(open('data/Advertising.csv'))
next(csvreader)  # ��� ����  (1��° ��)
x = []
y = []
for line in csvreader:
    x_i = [float(line[1]), float(line[2]), float(line[3])]
    y_i = float(line[4])
    x.append(x_i)
    y.append(y_i)
X = np.array(x)
Y = np.array(y)

# ���׽� ȸ�ͺм��� �����ϱ� ���� �������� �����մϴ�.
X_poly = []
for x_i in X:
    X_poly.append([
        x_i[0] * x_i[1] , # X_1^2
        x_i[0] , # X_2
        x_i[1] ** 10 , # X_2 * X_3
        x_i[2] ** 3 ,# X_3
        x_i[0] **2
    ])
print()

# X, Y�� 80:20���� �����ϴ�. 80%�� Ʈ���̴� ������, 20%�� �׽�Ʈ �������Դϴ�.
x_train, x_test, y_train, y_test = train_test_split(X_poly, Y, test_size=0.2, random_state=0)

# x_train, y_train�� ���� ���׽� ȸ�ͺм��� �����մϴ�.
lrmodel = LinearRegression()
lrmodel.fit(x_train, y_train)

beta_0 = lrmodel.coef_[0]
beta_1 = lrmodel.coef_[1]
beta_2 = lrmodel.coef_[2]
beta_3 = lrmodel.coef_[3]
beta_4 = lrmodel.coef_[4]
beta_5 = lrmodel.intercept_

print("beta_0: %f" % beta_0)
print("beta_1: %f" % beta_1)
print("beta_2: %f" % beta_2)
print("beta_3: %f" % beta_3)
print("beta_4: %f" % beta_4)
print("beta_5: %f" % beta_5)


#x_train�� ����, ���� ȸ�͸��� �������� ���ϰ�, �� ���� y_train �� ���̸� �̿��� MSE�� ���մϴ�.
predicted_y_train = lrmodel.predict(x_train)
mse_train = mean_squared_error(y_train, predicted_y_train)
print("MSE on train data: {}".format(mse_train))

# x_test�� ����, ���� ȸ�͸��� �������� ���ϰ�, �� ���� y_test �� ���̸� �̿��� MSE�� ���մϴ�. �� ���� 1 �̸��� �ǵ��� ���� ������ ���ϴ�.
predicted_y_test = lrmodel.predict(x_test)
mse_test = mean_squared_error(y_test, predicted_y_test)
print("MSE on test data: {}".format(mse_test))

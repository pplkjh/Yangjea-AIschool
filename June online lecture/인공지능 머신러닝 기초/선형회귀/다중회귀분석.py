import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import csv


'''
./data/Advertising.csv ���� �����͸� �о�, X�� Y�� ����ϴ�.

X�� (200, 3) �� shape�� ���� 2���� np.array,
Y�� (200,) �� shape�� ���� 1���� np.array���� �մϴ�.

X�� FB, TV, Newspaper column �� �ش��ϴ� �����͸� �����ؾ� �մϴ�.
Y�� Sales column �� �ش��ϴ� �����͸� �����ؾ� �մϴ�.
'''
csvreader = csv.reader(open('data/Advertising.csv'))

x = []
y = []

next(csvreader)  # ��� ����  (1��° ��)
for line in csvreader:
    x_i = [float(line[1]), float(line[2]), float(line[3])]
    y_i = float(line[4])
    x.append(x_i)
    y.append(y_i)
    
X = np.array(x)
Y = np.array(y)

lrmodel = LinearRegression()
lrmodel.fit(X, Y)

beta_0 = lrmodel.coef_[0]   #x1 (FB)�� ����
beta_1 = lrmodel.coef_[1]
beta_2 = lrmodel.coef_[2]
beta_3 = lrmodel.intercept_  #Y����

print("beta_0: %f" % beta_0)
print("beta_1: %f" % beta_1)
print("beta_2: %f" % beta_2)   # ��Ÿ2���� -�� ����: 1. �������� ������, ��Ÿ0��1 ��ŭ�� ū ������ ���� �ʴ´�.���� ������
print("beta_3: %f" % beta_3)

def expected_sales(fb, tv, newspaper, beta_0, beta_1, beta_2, beta_3):
    '''
    FB�� fb��ŭ, TV�� tv��ŭ, Newspaper�� newspaper ��ŭ�� ����� ����߰�,
    Ʈ���̴׵� ���� weight ���� beta_0, beta_1, beta_2, beta_3 �� ��
    ����Ǵ� Sales �� ���� ����մϴ�.
    '''
    #total_loss = np.sum((Y - ((beta_0 * fb) + (beta_1 * tv) + (beta_2 * newspaper) + beta_3)) ** 2)
    sales = (beta_0 * fb) + (beta_1 * tv) + (beta_2 * newspaper) + beta_3
    return sales

print("���� �Ǹŷ�: %f" % expected_sales(10, 12, 3, beta_0, beta_1, beta_2, beta_3))
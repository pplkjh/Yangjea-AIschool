import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import csv


'''
./data/Advertising.csv 에서 데이터를 읽어, X와 Y를 만듭니다.

X는 (200, 3) 의 shape을 가진 2차원 np.array,
Y는 (200,) 의 shape을 가진 1차원 np.array여야 합니다.

X는 FB, TV, Newspaper column 에 해당하는 데이터를 저장해야 합니다.
Y는 Sales column 에 해당하는 데이터를 저장해야 합니다.
'''
csvreader = csv.reader(open('data/Advertising.csv'))

x = []
y = []

next(csvreader)  # 헤더 생략  (1번째 열)
for line in csvreader:
    x_i = [float(line[1]), float(line[2]), float(line[3])]
    y_i = float(line[4])
    x.append(x_i)
    y.append(y_i)
    
X = np.array(x)
Y = np.array(y)

lrmodel = LinearRegression()
lrmodel.fit(X, Y)

beta_0 = lrmodel.coef_[0]   #x1 (FB)의 기울기
beta_1 = lrmodel.coef_[1]
beta_2 = lrmodel.coef_[2]
beta_3 = lrmodel.intercept_  #Y절편

print("beta_0: %f" % beta_0)
print("beta_1: %f" % beta_1)
print("beta_2: %f" % beta_2)   # 베타2값이 -인 이유: 1. 노이지한 데이터, 베타0과1 만큼의 큰 요인이 되지 않는다.모델이 가볍다
print("beta_3: %f" % beta_3)

def expected_sales(fb, tv, newspaper, beta_0, beta_1, beta_2, beta_3):
    '''
    FB에 fb만큼, TV에 tv만큼, Newspaper에 newspaper 만큼의 광고비를 사용했고,
    트레이닝된 모델의 weight 들이 beta_0, beta_1, beta_2, beta_3 일 때
    예상되는 Sales 의 양을 출력합니다.
    '''
    #total_loss = np.sum((Y - ((beta_0 * fb) + (beta_1 * tv) + (beta_2 * newspaper) + beta_3)) ** 2)
    sales = (beta_0 * fb) + (beta_1 * tv) + (beta_2 * newspaper) + beta_3
    return sales

print("예상 판매량: %f" % expected_sales(10, 12, 3, beta_0, beta_1, beta_2, beta_3))
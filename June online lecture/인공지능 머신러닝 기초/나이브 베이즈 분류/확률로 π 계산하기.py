import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import elice_utils

def main():
    plt.figure(figsize=(5,5))
    
    X = []
    Y = []
    
    # N을 10배씩 증가할 때 파이 값이 어떻게 변경되는지 확인해보세요.
    N = 1000
    
    for i in range(N):
        X.append(np.random.rand() * 2 - 1) 
        Y.append(np.random.rand() * 2 - 1)
    X = np.array(X)
    Y = np.array(Y)
    distance_from_zero = np.sqrt(X * X + Y * Y)
    is_inside_circle = distance_from_zero <= 1
    
    print("Estimated pi = %f" % (np.average(is_inside_circle) * 4)) #반지름이 1인 원의 넓이 --> πr^2 = π*1/4  ---> 이므로 4를 곱하면 π 값이 나온다
    
    plt.scatter(X, Y, c=is_inside_circle)
    plt.savefig('circle.png')
    elice_utils.send_image('circle.png')

if __name__ == "__main__":
    main()
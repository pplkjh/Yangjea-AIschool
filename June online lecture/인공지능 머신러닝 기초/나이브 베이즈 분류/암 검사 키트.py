def main():
    sensitivity = 0.8 #float(input())    # 암 보균자를 대상으로 키트가 양성일 확률 P(a1|b1)
    prior_prob = 0.004 #float(input())    # 실제로 암을 가지고 있을 확률
    false_alarm = 0.1 #float(input())   #실제로 암을 가지고 있지 않지만 양성으로 나올 확률

    print("%.2lf%%" % (100 * mammogram_test(sensitivity, prior_prob, false_alarm)))

def mammogram_test(sensitivity, prior_prob, false_alarm):
    # a0: 암 음성  a1:암 양성  b0: 실제 정상  b1: 실제 암
    p_a1_b1 = sensitivity # p(A = 1 | B = 1)   # 실제로 암을 가지고 있을때 암으로 진단할 확률  =0.8

    p_b1 = prior_prob    # p(B = 1)  # 암을 가지고 있을 확률   =0.004

    p_b0 = 1 - p_b1    # p(B = 0)    # 암을 가지고 있지 않을 확률   =0.996

    p_a1_b0 = false_alarm # p(A = 1|B = 0)  # 암을 가지고 있지 않지만 암으로 진단할 확률  =0.1
    
    # P(A = 1) = P(A = 1|B = 0)P(B = 0) + P(A = 1|B = 1)P(B = 1)
    p_a1 = p_a1_b0 * p_b0 + p_a1_b1 * p_b1    # p(A = 1) # 암이 진단될 확률  = 0.1028

    p_b1_a1 = p_a1_b1 * p_b1 / p_a1 # p(B = 1|A = 1)  # 암으로 진단을 받았을때 실제 암일 확률 = 0.0311

    return p_b1_a1

if __name__ == "__main__":
    main()

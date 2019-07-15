import tensorflow as tf

###########이 값은 바꾸지 마세요###########
output_weight = 0
output_bias = 0
###########################################

train_step = 500
learning_rate = 0.1

x_data = [1, 2, 3]
y_data = [1, 2, 3]
# W와 b에는 랜덤 값을 주고 아무 선이나 그리게 합니다.
W = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

# 데이터가 들어갈 placeholder를 만들어 줍니다
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 이 아무 선(가상의 선)을 hypothesis라고 변수를 주고
# 가상 선의 식을 적어줍니다.
# y = W^T * X + b
hypothesis = tf.add(X*W, b)

# cost function의 식입니다
# tf.reduce_mean은 평균을 구하는 함수
# tf.square는 제곱을 하는 함수
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 그레디언트 디센트라는 함수 최적화 알고리즘을 사용해 cost function을 최소화한다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('랜덤 초기화 값')
    print('weight : ', sess.run(W)[0])
    print('bias : ', sess.run(b)[0])
    print()
    for step in range(train_step):
        # 매번 train 변수를 통해 optimizer로 cost function을 최소화한다.
        sess.run(train, feed_dict = {X: x_data, Y: y_data})
        if step % 100 == 9:
            print(step)
            print('cost : ',sess.run(cost, feed_dict={X: x_data, Y: y_data}))
            print('weight : ', sess.run(W)[0])
            print('bias : ', sess.run(b)[0])
            print()
    
        output_weight = sess.run(W)[0]
        output_bias = sess.run(b)[0]
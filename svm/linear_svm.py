#!/usr/bin/env python
# -*- coding: utf-8 

# 本代码参考《tensorflow_cookbook》
# 本人做了详细解释，原版权归属原著，为清晰理解，代码顺序稍有调整
# 读者可打开print()方法，查看详细输出
# 本项目运行过程中，如有报错，是因为版本问题，根据错误提示，修改方法即可
# 根据提示，百度谷歌，不要着急，一点一点解决

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()
# matplotlib.use('Agg') # 在Linux下，保存图片，启动此配置

# 创建一个计算图，以后计算均在这个图上进行
sess = tf.Session()

######################### 准备数据集 开始 #########################
# 加载 iris 数据集
iris = datasets.load_iris()
# 查看一下 iris 的数据集
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
# print("iris.data:",iris.data)
# print("iris.target:",iris.target)
# print("iris.data.size:",len(iris.data))
# print("iris.target.size:",len(iris.target))

# iris.data 的 0、3位赋值给x_vals
x_vals = np.array([[x[0], x[3]] for x in iris.data])
# iris.target 的label经过判断后赋值给y_vals
y_vals = np.array([1 if y==0 else -1 for y in iris.target])
# print("x_vals:",x_vals)
# print("y_vals:",y_vals)

# 随机分割为训练集和测试集，由于随机分割，所以每次训练结果是不同的
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
# 训练集
x_vals_train = x_vals[train_indices]
y_vals_train = y_vals[train_indices]
# 测试集
x_vals_test = x_vals[test_indices]
y_vals_test = y_vals[test_indices]
######################### 准备数据集 结束 #########################

# 初始化占位符，可以理解为形参，此时没有数值
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 创建线性回归的变量，用以维护状态
A = tf.Variable(tf.random_normal(shape=[2,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# 模型输出，tf.subtract()：减法，tf.matmul()：乘法
model_output = tf.subtract(tf.matmul(x_data, A), b)

######################### 损失函数 开始 #########################
# 损失公式：Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
# 实现 max(0, 1-pred*actual)
# tf.reduce_mean()：均值，tf.maximum()：返回a，b最大值，tf.multiply()：点乘
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))

# alpha 的值
alpha = tf.constant([0.01])
# 计算 A 的L2 范数
l2_norm = tf.reduce_sum(tf.square(A))
# 实现 alpha * L2_norm(A)^2
alpha_l2 = tf.multiply(alpha, l2_norm)
# 损失函数
loss = tf.add(classification_term, alpha_l2)
######################### 损失函数 结束 #########################

######################### 预测与精准度 开始 #########################
# 将模型输出 标记为 prediction
prediction = tf.sign(model_output)
# 计算出精准度
# tf.equal()返回两个矩阵的True、False矩阵，维度同prediction
# tf.cast()指定数据类型映射
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))
######################### 预测与精准度 结束 #########################

######################### 训练方法 开始 #########################
# 优化器函数，梯度下降法
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)
######################### 训练方法 结束 #########################

# 初始化变量，tf变量初始化才可使用
# init = tf.initialize_all_variables() # 旧版本
init = tf.global_variables_initializer()
sess.run(init)

######################### 训练 开始 #########################
batch_size = 100 # 设置批量大小
loss_vec = [] # 记录损失变化，图形展示
train_accuracy = [] # 记录训练精准度变化，图形展示
test_accuracy = [] # 记录训测试准度变化，图形展示

# 训练循环控制,进行500次循环
for i in range(500):
    
    # 读取训练集，随机的
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    
    # 训练一次
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    
    # 训练的损失
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    
    # 训练精准度
    train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: np.transpose([y_vals_train])})
    train_accuracy.append(train_acc_temp)
    
    # 测试精准度
    test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_accuracy.append(test_acc_temp)
    
    # 每100次输出一次
    if (i+1)%100==0:
        print("########################### 训练中  ###########################")
        print('Step #' + str(i+1) + '\nA = ' + str(sess.run(A)) + '\nb = ' + str(sess.run(b)))
        print('Loss = ' + str(temp_loss))
print("########################### 训练结束  ###########################")
######################### 训练 结束 #########################

######################### 图形展示 开始  #########################
print("########################### 训练变化 数字输出  ###########################")
print("loss_vec:",loss_vec)
print("train_accuracy:",train_accuracy)
print("test_accuracy:",test_accuracy)

print("########################### 训练变化 图形展示  ###########################")
# 线性回归的系数
[[a1], [a2]] = sess.run(A) # a1,a2
[[b]] = sess.run(b) # b
slope = -a2/a1 # 斜率
y_intercept = b/a1 # 截距

# 获取所有训练样本数据的 第 2 位
# 从 x_vals 的每一项 d 中 读取 [1]，并组成 x1_vals
x1_vals = [d[1] for d in x_vals]

# 使用回归方程计算最佳匹配
best_fit = []
for i in x1_vals:
  best_fit.append(slope*i+y_intercept)
  
# print("x1_vals:",x1_vals)
# print("best_fit:",best_fit)

# setosa只是一个类别名词
# 是 setosa 类的 x、y坐标
setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i]==1]
setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i]==1]
# print("setosa_x:",setosa_x)
# print("setosa_y:",setosa_y)

# 不是 setosa 类的 x、y坐标
not_setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i]==-1]
not_setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i]==-1]
# print("not_setosa_x:",not_setosa_x)
# print("not_setosa_y:",not_setosa_y)

# 画 所有数据 和 最佳匹配线
# 是 setosa 类的点
plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
# 不是 setosa 类的点
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
# 最佳匹配线
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
# 设置 y轴 范围
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()
# plt.savefig('/home/tf/04svm/02linear_svm/Sepal_Length_vs_Pedal_Width.png') #在linux环境下会用到

# 画精准度
plt.plot(train_accuracy, 'k-', label='Training Accuracy')
plt.plot(test_accuracy, 'r--', label='Test Accuracy')
plt.title('Train and Test Set Accuracies')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
# plt.savefig('/home/tf/04svm/02linear_svm/Train_and_Test_Set_Accuracies.png')

# 画损失变化
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
# plt.savefig('/home/tf/04svm/02linear_svm/TLoss_per_Generation.png')

######################### 图形展示 结束  #########################



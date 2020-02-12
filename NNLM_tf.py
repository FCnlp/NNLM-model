# code by Tae Hwan Jung @graykode
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# 数据准备
sentences = ["i like dog", "i love coffee", "i hate milk"]
word_list = " ".join(sentences).split()
wordunique = list(set(word_list))
word_dict = dict(zip(wordunique, list(range(len(wordunique)))))
number_dict = dict(zip(list(range(len(wordunique))), wordunique))
n_class = len(wordunique)
print(word_dict)
print(number_dict)


def make_batch(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])
    return input_batch, target_batch


input_batch, target_batch = make_batch(sentences)
print(input_batch)
sentences = ["i like dog", "i love coffee", "i hate milk"]

# NNLM Parameter
n_step = 2  # number of steps ['i like', 'i love', 'i hate']
n_hidden = 2  # number of hidden units
X = tf.placeholder(tf.float32, [None, n_step, n_class])
Y = tf.placeholder(tf.float32, [None, n_class])
input = tf.reshape(X, shape=[-1, n_class * n_step])
w1 = tf.Variable(tf.random_normal([n_step * n_class, n_hidden]))
b1 = tf.Variable(tf.random_normal([n_hidden]))
h1 = tf.nn.tanh(tf.matmul(input, w1) + b1)

w2 = tf.Variable(tf.random_normal([n_hidden, n_class]))
b2 = tf.Variable(tf.random_normal([n_class]))

w3 = tf.Variable(tf.random_normal([n_step * n_class, n_class]))

output = tf.matmul(h1, w2) + b2 + tf.matmul(input, w3)
prediction = tf.argmax(output, 1)
# 定义loss 优化
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# Training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for epoch in range(5000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

# Predict
predict = sess.run([prediction], feed_dict={X: input_batch})
input = [sen.split()[:2] for sen in sentences]
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n] for n in predict[0]])

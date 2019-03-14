---
layout: post_layout
title: tensorflow学习
time: 2016年05月14日 星期六
published: true
excerpt_separator: "```"
---
## 简介

## 语法

### placeholder
    x = tf.placeholder(tf.float32, [None, 784])
x为二维向量，行数不定，列数为784，浮点类型。

### variable
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
w为784*10的二维向量，全0，b为10列的一维向量，全0.

### softmax
	y = tf.nn.softmax(tf.matmul(x, W) + b)
x与w相乘加b，然后求softmax。

### cross-entropy
<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msub>
    <mi>H</mi>
    <mrow class="MJX-TeXAtom-ORD">
      <msup>
        <mi>y</mi>
        <mo>&#x2032;</mo>
      </msup>
    </mrow>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>y</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mo>&#x2212;<!-- − --></mo>
  <munder>
    <mo>&#x2211;<!-- ∑ --></mo>
    <mi>i</mi>
  </munder>
  <msubsup>
    <mi>y</mi>
    <mi>i</mi>
    <mo>&#x2032;</mo>
  </msubsup>
  <mi>log</mi>
  <mo>&#x2061;<!-- ⁡ --></mo>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>y</mi>
    <mi>i</mi>
  </msub>
  <mo stretchy="false">)</mo>
</math>
即实际值*log（计算值）求和的负。

	y_ = tf.placeholder(tf.float32, [None, 10])
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	
### GradientDescentOptimizer
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
学习率为0.5，利用梯度下降最小化cross_entropy。

### 主体过程
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	for i in range(1000):
  		batch_xs, batch_ys = mnist.train.next_batch(100)
  		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
每次从mnist测试例中取100个学习。

### 评估
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
	
### CNN
#### convolution and pooling
	def conv2d(x, W):
  		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(x):
  		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    //32个5*5的卷积核
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32]) 
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	//64个5*5的卷积核，对32个输出
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	
#### densely connected layer
	//全连接层
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
#### dropout
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	//输出softmax
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.initialize_all_variables())
	for i in range(20000):
  		batch = mnist.train.next_batch(50)
  		if i%100 == 0:
   			train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    		print("step %d, training accuracy %g"%(i, train_accuracy))
  		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	print("test accuracy %g"%accuracy.eval(feed_dict={
    	x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
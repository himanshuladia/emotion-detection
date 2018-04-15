import numpy as np 
import tensorflow as tf 

test_batch_size = 100

testFeatures = np.load('test_set/test_features.npy')
testLabels = np.load('test_set/test_labels.npy')
cls_true = np.argmax(testLabels, axis=1)

sess = tf.Session()

saver = tf.train.import_meta_graph('/home/himanshu/Desktop/Machine Learning/Deep Learning Sentdex/Emotion Detection/model/model.meta')
saver.restore(sess,tf.train.latest_checkpoint('/home/himanshu/Desktop/Machine Learning/Deep Learning Sentdex/Emotion Detection/model/'))
graph = tf.get_default_graph()
y_pred_cls = graph.get_tensor_by_name("y_pred_cls:0")
x = graph.get_tensor_by_name("x:0")

def test_accuracy():
	num = testLabels.shape[0]
	cls_pred = np.zeros(shape=num, dtype=np.int)
	i = 0
	while i<num:
		j = min(i+test_batch_size,num)
		images = testFeatures[i:j,:]
		feed_dict = {x: images}
		cls_pred[i:j] = sess.run(y_pred_cls, feed_dict)
		i = j

	correct = (cls_true == cls_pred)
	correct_sum = correct.sum()
	acc = (correct_sum/num) * 100
	return acc

print(test_accuracy())
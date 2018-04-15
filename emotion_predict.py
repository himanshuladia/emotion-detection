import numpy as np
import tensorflow as tf
import cv2

img = cv2.imread("/images/randomhappy.jpg",0)
img = cv2.resize(img, (48,48))
img_flat = img.reshape([48*48])

emotion_dict = {0:'Angry', 1:'Disgusted', 2:'Fearful', 3:'Happy', 4:'Sad', 5:'Surprised', 6:'Neutral'}

with tf.Session() as sess:
	saver = tf.train.import_meta_graph('/model/model.meta')
	saver.restore(sess,tf.train.latest_checkpoint('./'))
	graph = tf.get_default_graph()

	x = graph.get_tensor_by_name("x:0")
	y_pred_cls = graph.get_tensor_by_name("y_pred_cls:0")

	feed_dict = {x: [img_flat]}
	emotion_cls = sess.run(y_pred_cls, feed_dict)
	
	emotion_cls_scalar = np.asscalar(emotion_cls)

	emotion = emotion_dict[emotion_cls_scalar]

	print("Are you {}?".format(emotion))
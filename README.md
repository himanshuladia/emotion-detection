# Human Facial Emotion Recognition

A convolution neural network with 3 convolutional layers, 2 fully connected layer is setup. 
It is used to train a model to detect the human facial emotions of the following categories 
Angry, Disgusted, Fearful, Happy, Sad, Surprised, Neutral.

The network is trained on a set of 28000 black and white human faces. A test set of 5000 images is used to determine the accuracy
of the model. With the current configuration and random initialization, the model gave 52% accuracy on the test. 
The tensorflow checkpoint for the corresponding trained model is saved in /model folder.

Concretely, most of the error came from recognizing non happy faces since the difference between say, a disgusted and a fear face is comparatively less
than that of a happy face.

# How to use?
1. Add the image to be recognized in the images folder.
2. Open emotion_predict.py in a text editor and edit line 5 to point to the new image.
3. Execute emotion_predict.py in terminal or IDE to see the output.

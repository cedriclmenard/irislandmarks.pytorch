import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def centerCropSquare(img, center, side=None, scaleWRTHeight=None):
    a = side is None
    b = scaleWRTHeight is None
    assert (not a and b) or (a and not b) # Python doesn't have "xor"... C'mon Python!
    half = 0
    if side is None:
        half = int(img.shape[0]*scaleWRTHeight/2)
    else:
        half = int(side/2)
        
    
    return img[(center[0] - half):(center[0] + half), (center[1] - half):(center[1] + half), :]

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="../mediapipe/mediapipe/models/iris_landmark.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on image
img = cv2.imread("test.jpg")
centerRight = [485, 332]
centerLeft = [479, 638]
img = centerCropSquare(img, centerRight, side=300) # 400 is 1200 (image size) * 64/192, as the detector takes a 64x64 box inside the 192 image
img = np.fliplr(img) # the detector is trained on the left eye only, hence the flip

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (64, 64))
input_data = np.expand_dims(img.astype(np.float32)/127.5 - 1.0, axis=0)
# input_data = np.expand_dims(img.astype(np.float32)/255.0, axis=0)
input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data_0 = interpreter.get_tensor(output_details[0]['index'])
# indices = output_data[:, 0:2] * 64.0
eyes = output_data_0
iris = interpreter.get_tensor(output_details[1]["index"])

# print(indices)

plt.imshow(img, zorder=1)
x, y = eyes[0,::3], eyes[0,1::3]
plt.scatter(x, y, zorder=2, s=1.0)

x, y = iris[0,::3], iris[0,1::3]
plt.scatter(x, y, zorder=2, s=1.0, c="r")
plt.show()

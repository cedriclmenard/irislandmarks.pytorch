import torch
import torch.nn as nn
from irislandmarks import IrisLandmarks
import matplotlib.pyplot as plt
import cv2
import numpy as np

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

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = IrisLandmarks().to(gpu)
net.load_weights("irislandmarks.pth")

img = cv2.imread("test.jpg")
centerRight = [485, 332]
centerLeft = [479, 638]
img = centerCropSquare(img, centerRight, side=400) # 400 is 1200 (image size) * 64/192, as the detector takes a 64x64 box inside the 192 image
plt.imshow(img)
plt.show()
# tl = [467, 284]
# br = [504, 397]
# w = br[1] - tl[1]
# h = br[0] - tl[0]
# w = int(w*2.3)
# h = int(h*2.3)
# tl[0] -= int(h/2)
# tl[1] -= int(w/2)
# br[0] = tl[0] + h
# br[1] = tl[1] + w
# img = img[tl[0]:br[0], tl[1]:br[1]]
# plt.imshow(img)
# plt.show()
# img = np.fliplr(img) # the detector is trained on the left eye only, hence the flip

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img = cv2.resize(img, (64, 64))

# test = net.predict_on_image(img)

eye_gpu, iris_gpu = net.predict_on_image(img)
eye = eye_gpu.cpu().numpy()
iris = iris_gpu.cpu().numpy()


plt.imshow(img, zorder=1)
x, y = eye[:, 0], eye[:, 1]
plt.scatter(x, y, zorder=2, s=1.0)
x, y = iris[:, 0], iris[:, 1]
plt.scatter(x, y, zorder=2, s=1.0, c='r')
plt.show()

# torch.onnx.export(
#     net, 
#     (torch.randn(1,3,64,64, device='cuda'), ), 
#     "irislandmarks.onnx",
#     input_names=("image", ),
#     output_names=("preds", "conf"),
#     opset_version=9
# )
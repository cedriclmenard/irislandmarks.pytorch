# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# # Convert TFLite model to PyTorch
# 
# This uses the model **face_detection_front.tflite** from [MediaPipe](https://github.com/google/mediapipe/tree/master/mediapipe/models).
# 
# Prerequisites:
# 
# 1) Clone the MediaPipe repo:
# 
# ```
# git clone https://github.com/google/mediapipe.git
# ```
# 
# 2) Install **flatbuffers**:
# 
# ```
# git clone https://github.com/google/flatbuffers.git
# cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
# make -j
# 
# cd flatbuffers/python
# python setup.py install
# ```
# 
# 3) Clone the TensorFlow repo. We only need this to get the FlatBuffers schema files (I guess you could just download [schema.fbs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs)).
# 
# ```
# git clone https://github.com/tensorflow/tensorflow.git
# ```
# 
# 4) Convert the schema files to Python files using **flatc**:
# 
# ```
# ./flatbuffers/flatc --python tensorflow/tensorflow/lite/schema/schema.fbs
# ```
# 
# Now we can use the Python FlatBuffer API to read the TFLite file!

# %%
# !git clone https://github.com/google/mediapipe.git
# !git clone https://github.com/google/flatbuffers.git
# !cd flatbuffers ; cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ; make -j
# !cd flatbuffers/python ; python setup.py install
# !git clone https://github.com/tensorflow/tensorflow.git
# !./flatbuffers/flatc --python tensorflow/tensorflow/lite/schema/schema.fbs


# Now restart this notebook

# %%
import os
import numpy as np
from collections import OrderedDict


# ## Get the weights from the TFLite file

# Load the TFLite model using the FlatBuffers library:

# %%
from tflite import Model

# taken from arcore pod
data = open("../mediapipe/mediapipe/models/iris_landmark.tflite", "rb").read()
model = Model.GetRootAsModel(data, 0)


# %%
subgraph = model.Subgraphs(0)
subgraph.Name()


# %%
def get_shape(tensor):
    return [tensor.Shape(i) for i in range(tensor.ShapeLength())]


# List all the tensors in the graph:

# %%
for i in range(0, subgraph.TensorsLength()):
    tensor = subgraph.Tensors(i)
    print("%3d %30s %d %2d %s" % (i, tensor.Name(), tensor.Type(), tensor.Buffer(), 
                                  get_shape(subgraph.Tensors(i))))


# Make a look-up table that lets us get the tensor index based on the tensor name:

# %%
tensor_dict = {(subgraph.Tensors(i).Name().decode("utf8")): i 
               for i in range(subgraph.TensorsLength())}


# Grab only the tensors that represent weights and biases.

# %%
parameters = {}
for i in range(subgraph.TensorsLength()):
    tensor = subgraph.Tensors(i)
    if tensor.Buffer() > 0:
        name = tensor.Name().decode("utf8")
        parameters[name] = tensor.Buffer()

len(parameters)


# The buffers are simply arrays of bytes. As the docs say,
# 
# > The data_buffer itself is an opaque container, with the assumption that the
# > target device is little-endian. In addition, all builtin operators assume
# > the memory is ordered such that if `shape` is [4, 3, 2], then index
# > [i, j, k] maps to `data_buffer[i*3*2 + j*2 + k]`.
# 
# For weights and biases, we need to interpret every 4 bytes as being as float. On my machine, the native byte ordering is already little-endian so we don't need to do anything special for that.

# %%
def get_weights(tensor_name):
    i = tensor_dict[tensor_name]
    tensor = subgraph.Tensors(i)
    buffer = tensor.Buffer()
    shape = get_shape(tensor)
    assert(tensor.Type() == 0)  # FLOAT32
    # tensor types are here: https://github.com/jackwish/tflite/blob/master/tflite/TensorType.py
    
    W = model.Buffers(buffer).DataAsNumpy()
    W = W.view(dtype=np.float32)
    W = W.reshape(shape)
    return W


# %%
W = get_weights("conv2d_1/Kernel")
b = get_weights("conv2d_1/Bias")
W.shape, b.shape


# Now we can get the weights for all the layers and copy them into our PyTorch model.

# ## Convert the weights to PyTorch format

# %%
import torch
import torch.nn as nn
from irislandmarks import IrisLandmarks


# %%
net = IrisLandmarks()


# %%
net


# %%
net(torch.randn(2,3,64,64))[0].shape


# Make a lookup table that maps the layer names between the two models. We're going to assume here that the tensors will be in the same order in both models. If not, we should get an error because shapes don't match.

# %%
probable_names = []
for i in range(0, subgraph.TensorsLength()):
    tensor = subgraph.Tensors(i)
    if tensor.Buffer() > 0 and tensor.Type() == 0:
        probable_names.append(tensor.Name().decode("utf-8"))
        
probable_names[:5]


# %%
len(probable_names)


# %%
from pprint import pprint


# %%
pprint(list(zip(probable_names, net.state_dict())))


# %%
len(net.state_dict()), len(probable_names)


# %%
convert = {}
i = 0
for name, params in net.state_dict().items():
    if i < 85:
        convert[name] = probable_names[i]
        i += 1


# %%
manual_mapping = {
    'split_eye.0.convs.1.weight':   'p_re_lu_21/Alpha',
    'split_eye.0.convs.2.weight':   'depthwise_conv2d_10/Kernel',
    'split_eye.0.convs.2.bias':     'depthwise_conv2d_10/Bias',
    'split_eye.0.convs.3.weight':   'conv2d_22/Kernel',
    'split_eye.0.convs.3.bias':     'conv2d_22/Bias',
    'split_eye.0.act.weight':       'p_re_lu_22/Alpha',
    'split_eye.1.convs.0.weight':   'conv2d_23/Kernel',
    'split_eye.1.convs.0.bias':     'conv2d_23/Bias',
    'split_eye.1.convs.1.weight':   'p_re_lu_23/Alpha',
    'split_eye.1.convs.2.weight':   'depthwise_conv2d_11/Kernel',
    'split_eye.1.convs.2.bias':     'depthwise_conv2d_11/Bias',
    'split_eye.1.convs.3.weight':   'conv2d_24/Kernel',
    'split_eye.1.convs.3.bias':     'conv2d_24/Bias',
    'split_eye.1.act.weight':       'p_re_lu_24/Alpha',
    'split_eye.2.convs.0.weight':   'conv2d_25/Kernel',
    'split_eye.2.convs.0.bias':     'conv2d_25/Bias',
    'split_eye.2.convs.1.weight':   'p_re_lu_25/Alpha',
    'split_eye.2.convs.2.weight':   'depthwise_conv2d_12/Kernel',
    'split_eye.2.convs.2.bias':     'depthwise_conv2d_12/Bias',
    'split_eye.2.convs.3.weight':   'conv2d_26/Kernel',
    'split_eye.2.convs.3.bias':     'conv2d_26/Bias',
    'split_eye.2.act.weight':       'p_re_lu_26/Alpha',
    'split_eye.3.convs.0.weight':   'conv2d_27/Kernel',
    'split_eye.3.convs.0.bias':     'conv2d_27/Bias',
    'split_eye.3.convs.1.weight':   'p_re_lu_27/Alpha',
    'split_eye.3.convs.2.weight':   'depthwise_conv2d_13/Kernel',
    'split_eye.3.convs.2.bias':     'depthwise_conv2d_13/Bias',
    'split_eye.3.convs.3.weight':   'conv2d_28/Kernel',
    'split_eye.3.convs.3.bias':     'conv2d_28/Bias',
    'split_eye.3.act.weight':       'p_re_lu_28/Alpha',
    'split_eye.4.convs.0.weight':   'conv2d_29/Kernel',
    'split_eye.4.convs.0.bias':     'conv2d_29/Bias',
    'split_eye.4.convs.1.weight':   'p_re_lu_29/Alpha',
    'split_eye.4.convs.2.weight':   'depthwise_conv2d_14/Kernel',
    'split_eye.4.convs.2.bias':     'depthwise_conv2d_14/Bias',
    'split_eye.4.convs.3.weight':   'conv2d_30/Kernel',
    'split_eye.4.convs.3.bias':     'conv2d_30/Bias',
    'split_eye.4.act.weight':       'p_re_lu_30/Alpha',
    'split_eye.5.convs.0.weight':   'conv2d_31/Kernel',
    'split_eye.5.convs.0.bias':     'conv2d_31/Bias',
    'split_eye.5.convs.1.weight':   'p_re_lu_31/Alpha',
    'split_eye.5.convs.2.weight':   'depthwise_conv2d_15/Kernel',
    'split_eye.5.convs.2.bias':     'depthwise_conv2d_15/Bias',
    'split_eye.5.convs.3.weight':   'conv2d_32/Kernel',
    'split_eye.5.convs.3.bias':     'conv2d_32/Bias',
    'split_eye.5.act.weight':       'p_re_lu_32/Alpha',
    'split_eye.6.convs.0.weight':   'conv2d_33/Kernel',
    'split_eye.6.convs.0.bias':     'conv2d_33/Bias',
    'split_eye.6.convs.1.weight':   'p_re_lu_33/Alpha',
    'split_eye.6.convs.2.weight':   'depthwise_conv2d_16/Kernel',
    'split_eye.6.convs.2.bias':     'depthwise_conv2d_16/Bias',
    'split_eye.6.convs.3.weight':   'conv2d_34/Kernel',
    'split_eye.6.convs.3.bias':     'conv2d_34/Bias',
    'split_eye.6.act.weight':       'p_re_lu_34/Alpha',
    'split_eye.7.convs.0.weight':   'conv2d_35/Kernel',
    'split_eye.7.convs.0.bias':     'conv2d_35/Bias',
    'split_eye.7.convs.1.weight':   'p_re_lu_35/Alpha',
    'split_eye.7.convs.2.weight':   'depthwise_conv2d_17/Kernel',
    'split_eye.7.convs.2.bias':     'depthwise_conv2d_17/Bias',
    'split_eye.7.convs.3.weight':   'conv2d_36/Kernel',
    'split_eye.7.convs.3.bias':     'conv2d_36/Bias',
    'split_eye.7.act.weight':       'p_re_lu_36/Alpha',
    'split_eye.8.weight':           'conv_eyes_contours_and_brows/Kernel',
    'split_eye.8.bias':             'conv_eyes_contours_and_brows/Bias',
    'split_iris.0.convs.0.weight':  'conv2d_37/Kernel',
    'split_iris.0.convs.0.bias':    'conv2d_37/Bias',
    'split_iris.0.convs.1.weight':  'p_re_lu_37/Alpha',
    'split_iris.0.convs.2.weight':  'depthwise_conv2d_18/Kernel',
    'split_iris.0.convs.2.bias':    'depthwise_conv2d_18/Bias',
    'split_iris.0.convs.3.weight':  'conv2d_38/Kernel',
    'split_iris.0.convs.3.bias':    'conv2d_38/Bias',
    'split_iris.0.act.weight':      'p_re_lu_38/Alpha',
    'split_iris.1.convs.0.weight':  'conv2d_39/Kernel',
    'split_iris.1.convs.0.bias':    'conv2d_39/Bias',
    'split_iris.1.convs.1.weight':  'p_re_lu_39/Alpha',
    'split_iris.1.convs.2.weight':  'depthwise_conv2d_19/Kernel',
    'split_iris.1.convs.2.bias':    'depthwise_conv2d_19/Bias',
    'split_iris.1.convs.3.weight':  'conv2d_40/Kernel',
    'split_iris.1.convs.3.bias':    'conv2d_40/Bias',
    'split_iris.1.act.weight':      'p_re_lu_40/Alpha',
    'split_iris.2.convs.0.weight':  'conv2d_41/Kernel',
    'split_iris.2.convs.0.bias':    'conv2d_41/Bias',
    'split_iris.2.convs.1.weight':  'p_re_lu_41/Alpha',
    'split_iris.2.convs.2.weight':  'depthwise_conv2d_20/Kernel',
    'split_iris.2.convs.2.bias':    'depthwise_conv2d_20/Bias',
    'split_iris.2.convs.3.weight':  'conv2d_42/Kernel',
    'split_iris.2.convs.3.bias':    'conv2d_42/Bias',
    'split_iris.2.act.weight':      'p_re_lu_42/Alpha',
    'split_iris.3.convs.0.weight':  'conv2d_43/Kernel',
    'split_iris.3.convs.0.bias':    'conv2d_43/Bias',
    'split_iris.3.convs.1.weight':  'p_re_lu_43/Alpha',
    'split_iris.3.convs.2.weight':  'depthwise_conv2d_21/Kernel',
    'split_iris.3.convs.2.bias':    'depthwise_conv2d_21/Bias',
    'split_iris.3.convs.3.weight':  'conv2d_44/Kernel',
    'split_iris.3.convs.3.bias':    'conv2d_44/Bias',
    'split_iris.3.act.weight':      'p_re_lu_44/Alpha',
    'split_iris.4.convs.0.weight':  'conv2d_45/Kernel',
    'split_iris.4.convs.0.bias':    'conv2d_45/Bias',
    'split_iris.4.convs.1.weight':  'p_re_lu_45/Alpha',
    'split_iris.4.convs.2.weight':  'depthwise_conv2d_22/Kernel',
    'split_iris.4.convs.2.bias':    'depthwise_conv2d_22/Bias',
    'split_iris.4.convs.3.weight':  'conv2d_46/Kernel',
    'split_iris.4.convs.3.bias':    'conv2d_46/Bias',
    'split_iris.4.act.weight':      'p_re_lu_46/Alpha',
    'split_iris.5.convs.0.weight':  'conv2d_47/Kernel',
    'split_iris.5.convs.0.bias':    'conv2d_47/Bias',
    'split_iris.5.convs.1.weight':  'p_re_lu_47/Alpha',
    'split_iris.5.convs.2.weight':  'depthwise_conv2d_23/Kernel',
    'split_iris.5.convs.2.bias':    'depthwise_conv2d_23/Bias',
    'split_iris.5.convs.3.weight':  'conv2d_48/Kernel',
    'split_iris.5.convs.3.bias':    'conv2d_48/Bias',
    'split_iris.5.act.weight':      'p_re_lu_48/Alpha',
    'split_iris.6.convs.0.weight':  'conv2d_49/Kernel',
    'split_iris.6.convs.0.bias':    'conv2d_49/Bias',
    'split_iris.6.convs.1.weight':  'p_re_lu_49/Alpha',
    'split_iris.6.convs.2.weight':  'depthwise_conv2d_24/Kernel',
    'split_iris.6.convs.2.bias':    'depthwise_conv2d_24/Bias',
    'split_iris.6.convs.3.weight':  'conv2d_50/Kernel',
    'split_iris.6.convs.3.bias':    'conv2d_50/Bias',
    'split_iris.6.act.weight':      'p_re_lu_50/Alpha',
    'split_iris.7.convs.0.weight':  'conv2d_51/Kernel',
    'split_iris.7.convs.0.bias':    'conv2d_51/Bias',
    'split_iris.7.convs.1.weight':  'p_re_lu_51/Alpha',
    'split_iris.7.convs.2.weight':  'depthwise_conv2d_25/Kernel',
    'split_iris.7.convs.2.bias':    'depthwise_conv2d_25/Bias',
    'split_iris.7.convs.3.weight':  'conv2d_52/Kernel',
    'split_iris.7.convs.3.bias':    'conv2d_52/Bias',
    'split_iris.7.act.weight':      'p_re_lu_52/Alpha',
    'split_iris.8.weight':          'conv_iris/Kernel',
    'split_iris.8.bias':            'conv_iris/Bias'
}
convert.update(manual_mapping)


# Copy the weights into the layers.
# 
# Note that the ordering of the weights is different between PyTorch and TFLite, so we need to transpose them.
# 
# Convolution weights:
# 
#     TFLite:  (out_channels, kernel_height, kernel_width, in_channels)
#     PyTorch: (out_channels, in_channels, kernel_height, kernel_width)
# 
# Depthwise convolution weights:
# 
#     TFLite:  (1, kernel_height, kernel_width, channels)
#     PyTorch: (channels, 1, kernel_height, kernel_width)
#     
# PReLU:
# 
#     TFLite:  (1, 1, num_channels)
#     PyTorch: (num_channels, )
# 

# %%
new_state_dict = OrderedDict()

for dst, src in convert.items():
    W = get_weights(src)
    print(dst, src, W.shape, net.state_dict()[dst].shape)

    if W.ndim == 4:
        if W.shape[0] == 1 and dst != "conf_head.4.weight":
            W = W.transpose((3, 0, 1, 2))  # depthwise conv
        else:
            W = W.transpose((0, 3, 1, 2))  # regular conv
    elif W.ndim == 3:
        W = W.reshape(-1)
    
    new_state_dict[dst] = torch.from_numpy(W)


# %%
net.load_state_dict(new_state_dict, strict=True)


# No errors? Then the conversion was successful!

# ## Save the checkpoint

# %%
torch.save(net.state_dict(), "irislandmarks.pth")


# %%




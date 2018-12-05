import onnx
from onnx_tf.backend import prepare

model = onnx.load('AutoEncoder.onnx')
AE_model = prepare(model)

print("success")
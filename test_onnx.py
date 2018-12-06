import onnx
from backend import prepare



agent = prepare(onnx.load('SpaceInvaders-v0.fskip7.160.tar.onnx'))
AE = prepare(onnx.load("AutoEncoder.onnx"))


print("success")




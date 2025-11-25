import onnx

model_path = "src/main/resources/titanic_random_forest.onnx"
model = onnx.load(model_path)

print(model.graph.input)

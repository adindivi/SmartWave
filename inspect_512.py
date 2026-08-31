import onnx

def inspect_512_model():
    model_path = "c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e_512.onnx"
    model = onnx.load(model_path)
    
    print("Last 5 nodes of 512 model:")
    for node in model.graph.node[-5:]:
        print(f"  Op: {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")

if __name__ == "__main__":
    inspect_512_model()

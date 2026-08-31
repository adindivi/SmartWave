import onnx

def inspect_527_normalization():
    model_path = "c:/Users/user/Desktop/SmartWave/models_official/smartwave_cnn10_e2e.onnx"
    model = onnx.load(model_path)
    
    print("Nodes related to Normalization in 527 model:")
    for node in model.graph.node:
        if node.op_type in ["ReduceL2", "Clip", "Expand", "Div"]:
            print(f"  Op: {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")

if __name__ == "__main__":
    inspect_527_normalization()

import os
import json
import numpy as np
from tensorflow import keras

def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ic50_model.h5")
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "src", "app", "model_weights.json")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print(f"Loading Keras model from {model_path}...")
    model = keras.models.load_model(model_path, compile=False)
    
    weights_dict = {}
    
    # We will extract weights for each layer in our model:
    # Layer 0: Dense(256)
    # Layer 1: BatchNormalization
    # Layer 2: Dropout (no weights)
    # Layer 3: Dense(64)
    # Layer 4: Dense(1)
    
    for i, layer in enumerate(model.layers):
        name = layer.name
        layer_type = type(layer).__name__
        print(f"Layer {i}: Name={name}, Type={layer_type}")
        
        weights = layer.get_weights()
        if not weights:
            print(f"  No weights for layer {name}")
            continue
            
        if layer_type == "Dense":
            kernel, bias = weights
            weights_dict[f"dense_{i}_kernel"] = kernel.tolist() # shape (input_dim, output_dim)
            weights_dict[f"dense_{i}_bias"] = bias.tolist() # shape (output_dim,)
            print(f"  Extracted Dense: kernel={kernel.shape}, bias={bias.shape}")
            
        elif layer_type == "BatchNormalization":
            # BatchNormalization weights: gamma, beta, moving_mean, moving_variance
            # In Keras, weights are: [gamma, beta, moving_mean, moving_variance]
            gamma, beta, moving_mean, moving_variance = weights
            epsilon = layer.epsilon if hasattr(layer, "epsilon") else 0.001
            
            # Precompute scale and offset to simplify JS calculation:
            # y = gamma * (x - mean) / sqrt(var + eps) + beta
            #   = x * [gamma / sqrt(var + eps)] + [beta - gamma * mean / sqrt(var + eps)]
            std = np.sqrt(moving_variance + epsilon)
            scale = gamma / std
            offset = beta - (gamma * moving_mean) / std
            
            weights_dict[f"bn_{i}_scale"] = scale.tolist()
            weights_dict[f"bn_{i}_offset"] = offset.tolist()
            print(f"  Extracted BatchNormalization: scale={scale.shape}, offset={offset.shape}")
            
    print(f"Saving weights to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(weights_dict, f)
    print("Weights exported successfully.")

if __name__ == "__main__":
    main()

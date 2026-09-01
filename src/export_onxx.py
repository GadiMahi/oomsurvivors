import torch
import sys
from pathlib import Path

# Connect to your local repository architecture
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import build_model

def export_to_onnx(checkpoint_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    state = torch.load(checkpoint_path, map_location=device)
    config = state.get("config", {})
    model_cfg = config.get("model", {})
    
    # Initialize the model with the exact same architecture
    model = build_model(
        model_cfg.get("name", "nafnet"),
        scale=config.get("dataset", {}).get("scale", 2),
        dim=model_cfg.get("dim", 48),
        levels=model_cfg.get("levels", 2),
        blocks=model_cfg.get("blocks", 2),
        middle_blocks=model_cfg.get("middle_blocks", 2),
        non_local=model_cfg.get("non_local", True),
        nl_heads=model_cfg.get("nl_heads", 4),
        nl_window_size=model_cfg.get("nl_window_size", 8)
    ).to(device)
    
    # Clean SWA "module." prefix if it exists
    sd = state.get("model", state)
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    
    # Dummy input for tracing. NAFNet expects single channel.
    dummy_input = torch.randn(1, 1, 128, 128, device=device)
    
    print(f"Exporting dynamic ONNX graph to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        opset_version=17, # Opset 17 is required to cleanly export Scaled Dot-Product Attention
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height_out", 3: "width_out"}
        }
    )
    print("✅ ONNX export complete.")

if __name__ == "__main__":
    export_to_onnx(
        "/kaggle/working/runs/nonlocal/swa_best_nafnet.pt", 
        "/kaggle/working/oomsurvivors/nafnet.onnx"
    )
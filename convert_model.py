from ultralytics import YOLO
import os

def convert_to_tflite():
    """Convert trained YOLO model to TFLite format."""
    print("\n=== Converting YOLO model to TFLite ===")
    
    # Check if trained model exists
    model_path = 'runs/detect/komodo_model/weights/best.pt'
    if not os.path.exists(model_path):
        print("Error: Trained model not found!")
        print("Please run train_model.py first to train the model.")
        return
    
    print("Loading YOLO model...")
    model = YOLO(model_path)
    
    print("\nConverting to TFLite format...")
    # Export to TFLite format
    model.export(format='tflite', 
                imgsz=640,
                half=False,  # Use FP16 quantization
                int8=False,  # Don't use INT8 quantization
                simplify=True,  # Simplify model
                opset=12)     # ONNX opset version
    
    # Create Flutter assets directory if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    
    # Copy the converted model to Flutter assets
    source_path = 'runs/detect/komodo_model/weights/best.tflite'
    target_path = 'assets/komodo_model.tflite'
    
    if os.path.exists(source_path):
        import shutil
        shutil.copy2(source_path, target_path)
        print(f"\nModel converted and copied to: {target_path}")
        print("\nNow update your pubspec.yaml to include the model:")
        print("\nflutter:")
        print("  assets:")
        print("    - assets/komodo_model.tflite")
    else:
        print("Error: Conversion failed!")

if __name__ == "__main__":
    convert_to_tflite() 
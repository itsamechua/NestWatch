from ultralytics import YOLO
import yaml
import os

def create_dataset_yaml():
    """Create the dataset configuration file."""
    dataset_path = r"C:\Users\itsamechua\Downloads\Komodo Dragon.v1i.yolov8"
    
    dataset_config = {
        'path': dataset_path,
        'train': 'train/images',  # Training images path
        'val': 'valid/images',    # Validation images path
        'test': 'test/images',    # Test images path
        
        # Classes
        'names': {
            0: 'komodo dragon'  # Class name
        }
    }
    
    # Save the configuration
    yaml_path = 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f)
    
    return yaml_path

def train_model():
    """Train the YOLO model on Komodo dragon dataset."""
    print("Setting up training configuration...")
    yaml_path = create_dataset_yaml()
    
    print("Loading base model...")
    model = YOLO('yolov8n.pt')  # Load the base YOLOv8 model
    
    print("\nStarting training...")
    print("This may take a while. The model will be saved automatically.")
    
    # Train the model with improved parameters
    results = model.train(
        data=yaml_path,
        epochs=100,         # Train for longer
        imgsz=640,         # Image size
        batch=16,          # Increased batch size
        patience=20,       # More patience for early stopping
        name='komodo_model',# Name for the results folder
        conf=0.3,          # Confidence threshold during training
        iou=0.5,          # IOU threshold during training
        augment=True,      # Use data augmentation
        verbose=True       # Show more training information
    )
    
    print("\nTraining completed!")
    print("The trained model is saved in 'runs/detect/komodo_model'")
    print("\nNow you can run detect_komodo.py to use the trained model!")

if __name__ == "__main__":
    train_model() 
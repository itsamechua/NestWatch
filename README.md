# Komodo Dragon Detection System

A real-time Komodo dragon detection system using YOLOv8 and computer vision. This system can detect Komodo dragons through a webcam feed and provide audio alerts when detections occur.

## Features

- Real-time Komodo dragon detection using webcam
- Audio alerts on detection
- Confidence threshold adjustment
- Detection shape analysis to reduce false positives
- Screenshot capture functionality
- FPS counter and performance metrics

## Prerequisites

- Python 3.11 (TensorFlow doesn't support Python 3.12+)
- Webcam
- CUDA-capable GPU (recommended for better performance)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/komodo-detection.git
cd komodo-detection
```

2. Create a virtual environment and activate it:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the base YOLOv8 model:
```bash
# This will be downloaded automatically when running the scripts
# but you can also download it manually from:
# https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8n.pt
```

## Usage

### Training the Model

1. Prepare your dataset in YOLOv8 format
2. Update `dataset.yaml` with your dataset paths
3. Run the training:
```bash
python train_model.py
```

### Running Detection

Run the detection script:
```bash
python detect_komodo.py
```

Controls during detection:
- Press 'q' or 'Esc' to quit
- Press 's' to save a screenshot
- Press '+' to increase confidence threshold
- Press '-' to decrease confidence threshold

### Converting the Model

To convert the trained model to TFLite format:
```bash
python convert_model.py
```

## Project Structure

- `detect_komodo.py`: Main detection script
- `train_model.py`: Model training script
- `convert_model.py`: Model conversion script
- `dataset.yaml`: Dataset configuration
- `alert.wav`: Sound alert file
- `requirements.txt`: Python dependencies

## Contributing

Feel free to open issues or submit pull requests for any improvements.

## License

[Your chosen license]

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- [Add any other acknowledgments] 

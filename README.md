
# YOLOv8-enhanced

## Overview
YOLOv8 is the a version of the YOLO (You Only Look Once) object detection model. It provides state-of-the-art performance for real-time object detection, instance segmentation, and image classification tasks.

## Features
- **Fast Detection**: Real-time inference on various hardware platforms
- **High Accuracy**: Improved mAP scores compared to previous versions
- **Easy Integration**: Simple Python API for quick implementation
- **Multi-Task Support**: Detection, segmentation, and classification in one framework

## Installation
```bash
pip install ultralytics
```

## Quick Start
```python
python install -e .
pip install wandb
wandb login
pip install ultralytics
yolo settings wandb=True
pip install timm
```


## Requirements
- Python 3.8+
- PyTorch 1.7+
- OpenCV
- cuda (optional for GPU support)

## Documentation
For detailed documentation, visit [Ultralytics YOLOv8](https://docs.ultralytics.com)

## License
YOLOv8 is released under the AGPL-3.0 License.

## Modifications
This repository includes custom enhancements to YOLOv8:
- **ACmix Module**: Advanced feature mixing for improved detection
- **CLLADetect**: Enhanced detection head for better performance
- **ESEMB & FASTER**: Efficient modules for speed and accuracy improvements
- **Custom Loss Functions**: Wasserstein loss for better training dynamics
- **P2 Detection**: Small object detection support at P2 scale

## Ablation Studies
Comprehensive ablation experiments have been conducted to validate the effectiveness of each module.

## Visualization
Run `python train.py` with custom modules to train the model and visualize results.
```bash
python train.py
```
Run `detectWindow.py` to visualize detection results with a graphical interface:
```bash
python detectWindow.py
```

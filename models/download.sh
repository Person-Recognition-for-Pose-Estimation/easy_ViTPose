#! /usr/bin/env bash
DIR=$(dirname "$0")
wget -O $DIR/vitpose-s-coco.onnx https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-s-coco.onnx
wget -O $DIR/yolov8n.pt https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt

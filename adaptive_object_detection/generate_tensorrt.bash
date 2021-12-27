#!/bin/bash
# This script generates a TensorRT engine from pretrained Tensorflow SSD Mobilenet v2 COCO model.

echo $relative_path
if [ $# -eq 2 ]
then
	pb_file=$1
	adaptive_model=1
	num_classes=$2

else 
	if [ ! -f "$relative_path/detectors/data/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"  ]; then
		echo "downloading ssd_mobilenet_v2_coco_2018_03_29 ..."
		wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz -P $relative_path/detectors/data/
	fi
	tar -xvf $relative_path/detectors/data/ssd_mobilenet_v2_coco_2018_03_29.tar.gz --no-same-owner -C $relative_path/detectors/data/ 
	pb_file="$relative_path/detectors/data/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
	adaptive_model=0
	num_classes=90

fi


echo "************  Generating TensorRT from: $pb_file  **************"
python3 $relative_path/exporters/trt_exporter.py --pb_file $pb_file --out_dir $relative_path/detectors/data/ --neuralet_adaptive_model $adaptive_model --num_classes $num_classes
#mv $relative_path/detectors/data/frozen_inference_graph.bin $relative_path/detectors/data/

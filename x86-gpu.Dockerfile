FROM nvcr.io/nvidia/tensorflow:21.12-tf2-py3

VOLUME  /repo
WORKDIR /repo

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip setuptools==41.0.0 && pip install opencv-python wget
RUN pip install streamlit

#ENV PYTHONPATH=/repo:/repo/adaptive_object_detection

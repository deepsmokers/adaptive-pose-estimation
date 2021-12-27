import imghdr
import os
import pathlib
import uuid
from dataclasses import dataclass

import cv2 as cv
import numpy as np
import streamlit as st

from adaptive_object_detection.utils.parse_label_map import create_category_index_dict
from tools.visualization_utils import visualize_poses

SIDEBAR_OPTION_PROJECT_DETAIL = "Show Project Detail"
SIDEBAR_OPTION_DEMO_IMAGE = "Select a Demo Image"
SIDEBAR_OPTION_UPLOAD_IMAGE = "Upload an Image"
CURRENT_PATH = pathlib.Path(__file__).parent
TEMP_IMAGE_PATH = pathlib.Path("/tmp/post-estimation")
DEMO_IMAGES_PATH = CURRENT_PATH.joinpath("demo_images")
SIDEBAR_OPTIONS = [
    SIDEBAR_OPTION_PROJECT_DETAIL,
    SIDEBAR_OPTION_DEMO_IMAGE,
    SIDEBAR_OPTION_UPLOAD_IMAGE,
]


@dataclass
class InferenceModel:
    config: str
    input_image_dir: str
    output_image_dir: str


@st.cache(allow_output_mutation=True)
def load_model():
    from models.x86_pose_estimator import X86PoseEstimator
    label_map_file = "adaptive_object_detection/utils/mscoco_label_map.pbtxt"
    label_map = create_category_index_dict(label_map_file)
    pose_estimator = X86PoseEstimator(detector_thresh=0.1,
                                      detector_input_size=(300, 300),
                                      pose_input_size=(256, 192),
                                      heatmap_size=(64, 48))
    pose_estimator.load_model(None, detector_label_map=label_map)
    return pose_estimator


def inference_image(image):
    left_column, right_column = st.columns(2)
    l_col, r_col = left_column.image(image, caption="Input"), right_column.image(image, caption="Output")

    with st.spinner('Wait for it...'):
        input_image_path = TEMP_IMAGE_PATH.joinpath(str(uuid.uuid4().hex))
        if not os.path.exists(input_image_path):
            os.makedirs(input_image_path, exist_ok=True)

        with open(input_image_path.joinpath(image.name), mode="wb") as f:
            f.write(image.read())

        image_path = str(input_image_path.joinpath(image.name))
        cv_image = cv.imread(image_path)
        height, width, channels = cv_image.shape
        if np.shape(cv_image) != ():
            pose_estimator = load_model()
            out_frame = cv.resize(cv_image, (width, height))
            preprocessed_image = pose_estimator.preprocess(cv_image)
            result_raw = pose_estimator.inference(preprocessed_image)
            result = pose_estimator.post_process(*result_raw)
            if result is not None:
                out_frame = visualize_poses(out_frame, result, (300, 300))
                rgb_image = cv.cvtColor(out_frame, cv.COLOR_BGR2RGB)

                r_col.image(rgb_image, caption="Output")


def show_demo_image(img: str):
    input_image_path = DEMO_IMAGES_PATH.joinpath(img).resolve()

    output_image_path = DEMO_IMAGES_PATH.joinpath(".".join(img.split(".")[:-1]) + "_output." + img.split(".")[-1]).resolve()

    left_column, right_column = st.columns(2)

    left_column.image(str(input_image_path), caption="Input", use_column_width="always")
    right_column.image(str(output_image_path), caption="Output")


def demo_images_container():
    st.sidebar.write(" ------ ")

    photos = []
    for file in os.listdir(DEMO_IMAGES_PATH):
        if "output" not in file:
            filepath = DEMO_IMAGES_PATH.joinpath(file)
            if imghdr.what(filepath) is not None:
                photos.append(file)

    photos.sort()

    inference_pressed = False
    option = None

    if photos:
        option = st.sidebar.selectbox('Please select a sample image, then click inference', photos)
        inference_pressed = st.sidebar.button('Inference')

    if inference_pressed:
        show_demo_image(option)


def upload_image_container():
    st.sidebar.info('PRIVACY POLICY: uploaded images are never saved or stored. They are held entirely within memory for prediction \
            and discarded after the final results are displayed. ')
    upload_txt = st.sidebar.empty()
    f = st.sidebar.file_uploader("Please Select to Upload an Image", type=['png', 'jpg', 'jpeg'])
    if f is not None:
        if f.size > 0:
            inference_pressed = st.sidebar.button('Inference')
            if inference_pressed:
                inference_image(f)
        else:
            upload_txt.error("Choose another file")


def main():
    st.set_page_config(page_title="FaceMask", layout="wide")
    st.markdown("<h1 style='text-align: center; color: white;'>Welcome to TinyPose</h1>", unsafe_allow_html=True)
    st.markdown("***")

    app_mode = st.sidebar.selectbox("Please select from the following", SIDEBAR_OPTIONS)

    if app_mode == SIDEBAR_OPTION_PROJECT_DETAIL:
        pass
    elif app_mode == SIDEBAR_OPTION_DEMO_IMAGE:
        demo_images_container()
    elif app_mode == SIDEBAR_OPTION_UPLOAD_IMAGE:
        upload_image_container()
    else:
        raise ValueError("Selected side bar option not available")


main()

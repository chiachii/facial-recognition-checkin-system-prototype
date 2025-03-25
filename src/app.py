import streamlit as st
import cv2
import numpy as np
import pickle
import yaml
from PIL import Image
import argparse
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os
from utils import get_face_embedding, predict_user

# Define the models
detector = MTCNN()
embedder = FaceNet()

# Function: face detection (MTCNN) + image resizing
def get_face_pixel(img, show_error_info=True):
    # Facial dectection
    is_face = detector.detect_faces(img)

    if not is_face:
        if show_error_info:
            st.error('Error: No face found, try another picture')
        return None, None
    else:
        # Take the face of first person
        x1, y1, width, height = is_face[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        face = img[y1:y2, x1:x2]
        face_resized = cv2.resize(face, (160, 160))
        return [x1, x2, y1, y2], face_resized
    


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db_path', default='../dataset/face_db.pkl', type=str)
    args = parser.parse_args()
    # Load the face database
    with open(args.db_path, 'rb') as f:
        face_db = pickle.load(f)
    
    # Settings
    st.set_page_config(layout='wide')
    # Config
    cfg = yaml.load(open('../config.yaml', 'r'), Loader=yaml.FullLoader)
    PICTURE_PROMPT = cfg['INFO']['PICTURE_PROMPT']
    WEBCAM_PROMPT = cfg['INFO']['WEBCAM_PROMPT']
    THRESHOLD_PROMPT = cfg['INFO']['THRESHOLD_PROMPT']
    st.sidebar.title('Settings')

    # Create a menu bar
    menu = ['Picture', 'Webcam']
    choice = st.sidebar.selectbox('Input Type', menu)
    # Put slide to adjust threshold
    THRESHOLD = st.sidebar.slider('Threshold', 0.0, 1.0, 0.7, 0.01)
    st.sidebar.info(THRESHOLD_PROMPT)
    
    # Information
    st.sidebar.title('Who are you?')
    st.sidebar.write('※ Checkin Successfully (Green) / Failed (Red)')
    name_container = st.sidebar.empty()
    name_container.info('Name: ')
    dist_container = st.sidebar.empty()
    dist_container.info('Dist: ')
    if choice == 'Picture':
        st.title('Facial Recognition Checkin System (Prototype)')
        st.write(PICTURE_PROMPT)
        uploaded_files = st.file_uploader('Upload', type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
        if len(uploaded_files) != 0:
            # Facial recognition for uploaded image
            for f in uploaded_files:
                # Load the images
                im = Image.open(f)
                im.save('temp.jpg')
                img = cv2.imread('temp.jpg')
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Face detection
                box, face_pixel = get_face_pixel(img_rgb)
                if face_pixel is not None:
                    pred_user_name, dist = predict_user(test_image=face_pixel[np.newaxis, :, :, :], 
                                                        user_reg=face_db, 
                                                        threshold=THRESHOLD)
                    pred_user_name = pred_user_name.replace('_', ' ')
                    if pred_user_name != 'Unknown': # TODO: how to solve when making wrong recognition?
                        name_container.success(f'Name: {pred_user_name}')
                        dist_container.success(f'Dist: {dist:>.3f}')
                    else:
                        name_container.error(f'Name: {pred_user_name}')
                        dist_container.error(f'Dist: {dist:>.3f}')
                    
                    # Show the results
                    st.markdown(f'### Recognition Result → {pred_user_name}', True)
                    [x1, x2, y1, y2] = box
                    cv2.rectangle(img_rgb, (x1, y1),(x2, y2), (0, 255, 0), 3)
                    cv2.putText(img_rgb, pred_user_name, (x1-20, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    st.image(img_rgb)
                    os.remove('temp.jpg')

    elif choice == 'Webcam':
        st.title('Facial Recognition Checkin System (Prototype)')
        st.write(WEBCAM_PROMPT)
        # Camera Settings
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        FRAME_WINDOW = st.image([])

        while True:
            ret, frame = cam.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                st.error('Failed to capture frame from camera')
                st.stop()
            
            try: # If there are faces, this process make more precision
                box, face_pixel = get_face_pixel(frame_rgb, False)
                pred_user_name, dist = predict_user(test_image=face_pixel[np.newaxis, :, :, :], 
                                                    user_reg=face_db, 
                                                    threshold=THRESHOLD)
            except:
                pred_user_name, dist = predict_user(test_image=frame_rgb[np.newaxis, :, :, :], 
                                                    user_reg=face_db, 
                                                    threshold=THRESHOLD)
            pred_user_name = pred_user_name.replace('_', ' ')
            if pred_user_name != 'Unknown': # TODO: how to solve when making wrong recognition?
                name_container.success(f'Name: {pred_user_name}')
                dist_container.success(f'Dist: {dist:>.3f}')
            else:
                name_container.error(f'Name: {pred_user_name}')
                dist_container.error(f'Dist: {dist:>.3f}')

            # Show the result:
            if box is not None:
                [x1, x2, y1, y2] = box
                cv2.rectangle(frame_rgb, (x1, y1),(x2, y2), (0, 255, 0), 3)
                cv2.putText(frame_rgb, pred_user_name, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            FRAME_WINDOW.image(frame_rgb)

if __name__ == '__main__':
    main()
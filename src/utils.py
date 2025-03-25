import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
import numpy as np
import os

# Define the models
detector = MTCNN()
embedder = FaceNet()

# Function: face detection (MTCNN) + image resizing
def lfw_dataset(img_path):
    # Load the image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Facial dectection
    is_face = detector.detect_faces(img_rgb)

    if not is_face:
        # print('[Error] No face found, try another picture!')
        return 
    else:
        # Take the face of first person
        x1, y1, width, height = is_face[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        face = img_rgb[y1:y2, x1:x2]
        face_resized = cv2.resize(face, (160, 160))
        return face_resized

# Function: face recognition (FaceNet)
def get_face_embedding(imgs):
    # Compute average face embedding
    face_embs = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        temp_emb = embedder.embeddings([img])[0]
        face_embs.append(temp_emb) 
    avg_face_emb = np.mean(face_embs, axis=0)
    return avg_face_emb

# Function: load images based on `user_name`
def get_face_imgs(user_name, dir='../dataset/lfw/lfw-deepfunneled/'):
    dir += f'{user_name}/'
    face_imgs = []
    for i in range(1, len(os.listdir(dir))+1):
        img_path = dir + f'{user_name}_{i:04}.jpg'
        img = lfw_dataset(img_path)
        if img is not None:
            face_imgs.append(img)
    return np.array(face_imgs)

# Function: prediction for new image
def predict_user(test_image, user_reg, threshold=0.7):
    curr_emb = get_face_embedding(test_image)
    if curr_emb is None:
        return 'Unknown'
    
    min_dist = float('inf')
    most_simi_user_name = 'Unknown'
    for user_name, face_emb in user_reg.items():
        dist = np.linalg.norm(curr_emb - face_emb)
        if dist < min_dist:
            min_dist = dist
            most_simi_user_name = user_name
    if min_dist > threshold:
        return 'Unknown', min_dist
    return most_simi_user_name, min_dist
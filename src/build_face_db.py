import numpy as np
import pandas as pd
import cv2
import pickle
import os
import argparse
from utils import *

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', default='../dataset/lfw/lfw-deepfunneled/', type=str)
    parser.add_argument('--db_path', default='../dataset/face_db.pkl', type=str)
    parser.add_argument('--add_user', default=None, type=str)
    parser.add_argument('--init', default=False, type=bool)
    args = parser.parse_args()
    
    if args.init:
        face_db = {}
        for user_name in os.listdir(args.dir):
            user_imgs = get_face_imgs(user_name=user_name, dir=args.dir)
            face_db[user_name] = get_face_embedding(user_imgs)
        with open(args.db_path, 'wb') as f:
            pickle.dump(face_db, f)
    
    if args.add_user is not None:
        user_imgs = get_face_imgs(user_name=args.add_user, dir=args.dir)
        user_emb = get_face_embedding(user_imgs)
        
        # Write into the face database
        if os.path.exists(args.db_path):
            with open(args.db_path, 'rb') as f:
                face_db = pickle.load(f)
        else:
            face_db = {}
        face_db[args.add_user] = user_emb
        with open(args.db_path, 'wb') as f:
            pickle.dump(face_db, f)

if __name__ == '__main__':
    main()
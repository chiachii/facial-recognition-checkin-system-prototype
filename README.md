# Facial Recognition Check-in System Prototype
# Demo
![alt text](assets/picture.png) 
![alt text](assets/webcam.png)

# Introduction
This project is a simple check-in system using Python, pretrained MTCNN + FaceNet, and Streamlit that supports recognition on the uploaded images/webcam.
The app built a face database based on the LFW dataset and achieved 96.28% accuracy (shown as `src/demo_on_lfw_dataset.ipynb`).

## Getting Started
1. Clone the repository.
```bash
git clone https://github.com/chiachii/facial-recognition-checkin-system-prototype.git
cd facial-recognition-checkin-system-prototype/src/
```
2. Build a new environment & Install all the required Python packages.
```bash
conda env -n face_checkin
conda activate face_checkin
pip install -r requirements.txt
```
3. Build a facial embedding database based on LFW dataset. If you would like to add user, you can just add at least one picture into `dataset/lfw/lfw-deepfunneled/{user_name}/` (nameed as `{user_name}_0001.jpg`). (Build the database by whole dataset: `--init=True` / Add user: `--add_user={user_name}`)
```bash
python build_face_db.py --init=True --add_user={user_name}
```
4. To run the app.
```bash
streamlit run app.py
```

## Limitation
This project cannot be applied to multi-face recognition, in which case MTCNN may make a wrong detection causing FaceNet to make a wrong recognition as well.
![](assets/limitation.png)

## Reference
- Chung Tien Dat (datct00): [Face-recognition-app-using-Streamlit](https://github.com/datct00/Face-recognition-app-using-Streamlit)
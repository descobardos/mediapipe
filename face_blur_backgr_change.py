import streamlit as st
import shutil
import cv2
import mediapipe as mp
import os
from os import path
import numpy as np
from PIL import Image
import uuid


st.title("Difuminado de Rostro & Cambio de Background")
try:
    os.mkdir("temp")
except:
    pass
for i in os.listdir("./temp/"):
    try:
        os.remove(os.remove(f"./temp/{i}"))
    except:
        pass
input_file_path = ""
uploaded_file = st.file_uploader("Subir Archivo a difuminar", type=["mp4"])
if uploaded_file is not None:
    with open(f"./temp/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    input_file_path = f"./temp/{uploaded_file.name}"

folder_path = st.text_input(label="Ingresa el nombre del directorio para guardar el archivo procesado:", placeholder="ej:  C:/dekstop/ratatoille/sorprendeme")
flag = path.exists(folder_path)

class FaceDetector:
    def __init__(self, minDetectionCon=0.45, pixel_size=0.05):
        self.pixel_size = pixel_size
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.faceDetection.process(imgRGB)
        if self.result.detections:
            for id, detection in enumerate(self.result.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = (
                    int(bboxC.xmin * iw) - 40,
                    int(bboxC.ymin * ih) - 150,
                    int(bboxC.width * iw) + 80,
                    int(bboxC.height * ih) + 200,
                )
                x, y, w, h = bbox
                x1, y1 = x + w, y + h
                try:
                    face_img = img[y:y1, x:x1].copy()
                    face_img = cv2.resize(
                        face_img,
                        dsize=None,
                        fx=self.pixel_size,
                        fy=self.pixel_size,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    face_img = cv2.resize(
                        face_img,
                        dsize=(x1 - x, y1 - y),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    img[y:y1, x:x1] = face_img
                except:
                    pass
        return img

class BackgroundRemove:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection = 1)
    def remove_background(self ,img, option,original_background):
        if option == "color":
            BG_COLOR = (0, 255, 0)
        
        bg_image = None
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        self.results = self.selfie_segmentation.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        condition = np.stack((self.results.segmentation_mask,) * 3, axis =-1) > 0.1
        if bg_image is None:
            if option =="Blur":
                bg_image = cv2.GaussianBlur(image,(55,55),0)
            elif option == "Color":
                BG_COLOR = (0, 255, 0)
                bg_image = np.zeros(image.shape, dtype=np.uint8)
                bg_image[:] = BG_COLOR
            elif option =="Cargar Imagen Background":
                ih, iw, ic = image.shape
                bg_image = cv2.cvtColor(original_background, cv2.COLOR_RGB2BGR)
                # bg_image = original_background
                bg_image = cv2.resize(bg_image, (iw,ih))
                # bg_image = cv2.GaussianBlur(bg_image,(55,55),0)
        output_image = np.where(condition, image, bg_image)
        return output_image

  
def main(detectionConfidence, blur_size, flip_the_video, option, original_background=None, enable_face_blur=False, background_change=False):
    global folder_path
    global input_file_path
    FRAME_WINDOW = st.image([])
    input_file = input_file_path
    cap = cv2.VideoCapture(input_file)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    size = (width, height)
    file_name = "./temp/output.mp4"
    if folder_path.endswith("/"):
        export_file_path = f"{folder_path}"
    else:
        export_file_path = f"{folder_path}/"

    var1 = os.system(f'ffmpeg -i {input_file} "./temp/audio.mp3"')

    if var1 == 0:
        print("audio extraido")
    # codec = cv2.VideoWriter_fourcc(*"mpeg")
    codec = cv2.VideoWriter_fourcc(*"MP4V")
    video_output = cv2.VideoWriter(file_name, codec, framerate, size, True)

    background_remove_object = BackgroundRemove()

    detector = FaceDetector(minDetectionCon = float(detectionConfidence / 100), pixel_size = float((blur_size)/1000))
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    while ret:
        success, img = cap.read()
        if flip_the_video == "Yes":
            img = cv2.flip(img, 1)
        elif flip_the_video == "No":
            pass
        try:
            if background_change:
                img = background_remove_object.remove_background(img, option, original_background)
            if enable_face_blur:
                img = detector.findFaces(img, True)
        except Exception as e:
            print(f"error is {e}")
        if img is None:
            break

        video_output.write(img)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame) 
    video_output.release()
    cap.release()
    audio_file = "./temp/audio.mp3"
    blur_video = "./temp/output.mp4"
    os.system(
        f"ffmpeg -i {file_name} -i {audio_file} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {blur_video}"
    )
    rename_file_name = f"{export_file_path}output_" + input_file.split("/")[-1]
    try:
        os.remove(rename_file_name)
    except:
        pass
    # try:
    #     shutil.copy(blur_video, rename_file_name)
    # except:
    unique_filename = str(uuid.uuid4())
    rename_file_name = f"{rename_file_name.split('.mp4')[0]}_{unique_filename}.mp4"
    shutil.copy(blur_video, rename_file_name)


if __name__ == "__main__":
    detectionConfidence = st.slider("Face Detection Confidence")
    blur_size = st.slider("Tamaño del pixel difuminado sobre el rostro")
    flip_the_video = st.selectbox("El video esta en forma horizontal?  ", ("Yes", "No"))

    enable_face_blur = st.checkbox("Difuminar Rostro")
    face_blocker = False
    background_blocker = False
    if enable_face_blur:
        face_blocker = True
    
    background_change = st.checkbox("Cambio de Background")
    if background_change:
        background_blocker = True

    option = None
    original_background = None
    if background_change:
        background_format = st.selectbox("Seleccione tipo Background ",("Color", "Blur", "Cargar Imagen Background"))
        if background_format == "Cargar Imagen Background":
            img_file_buffer = st.file_uploader("Subir imagen definida para background", type=["png", "jpg", "jpeg"])
            if img_file_buffer is not None:
                image = Image.open(img_file_buffer)
                img_array = np.array(image) # si se requiere pasar a OpenCV
                original_background = img_array
        option = background_format
   
    if st.button("Iniciar Proceso..."):
        if flag:
            main(detectionConfidence, blur_size, flip_the_video, option, original_background, enable_face_blur = face_blocker, background_change = background_blocker)
            st.markdown(f"### Difuminado de rostro finalizado, verifique su .mp4 en la carpeta indicada anteriormente...")

            for i in os.listdir("./temp/"):
                try:
                    os.remove(os.remove(f"./temp/{i}"))
                except:
                    pass
        else:
            st.error("Carpeta a exportar el archivo NO Existe.")
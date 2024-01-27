import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, minDetectionCon=0.45):
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
                # añadir algunos valores para aumentar el box size
                bbox = (
                    #cambiar valores para cambiar el area de blur box
                            int(bboxC.xmin * iw) - 40,
                            int(bboxC.ymin * ih) - 100,
                            int(bboxC.width * iw) + 80,
                            int(bboxC.height * ih) + 150,
                )
                #cambiar valores de ser requerido para box size
                x, y, w, h = bbox
                x1, y1 = x + w, y + h
                try:
                    face_img = img[y:y1, x:x1].copy()
                    face_img = cv2.resize(
                        face_img,
                        dsize = None,
                        fx = 0.05, #disminuir el valor para un mayor tamaño de pixel difuminado
                        fy = 0.05,  #disminuir el valor para un mayor tamaño de pixel difuminado
                        interpolation = cv2.INTER_NEAREST,
                    )
                    face_img = cv2.resize(
                        face_img,
                        dsize = (x1 - x, y1 - y),
                        interpolation = cv2.INTER_NEAREST,
                    )
                    img[y:y1, x:x1] = face_img
                except Exception as e:
                    print(e)
                    pass
        return img

def main():
    # cap=cv2.VideoCapture("videos/2.mp4")
    cap = cv2.VideoCapture(0)
    pTime = 0

    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findFaces(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(
            img,
            f"FPS : {int(fps)}",
            (20, 70),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
#import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import tkinter
from tkinter import messagebox
import smtplib

root = tkinter.Tk()
root.withdraw()

model = load_model('face_mask_detection\face_mask_detection_system.h5')
face_classifier = cv2.CascadeClassifier('face_mask_detection\haarcascade_frontalface_default.xml')

vid_capture = cv2.VideoCapture(0)

SUBJECT = "A Person Detected Without Mask"
TEXT = "One Visitor violated Face Mask Policy. See in the camera to recognize user. A Person has been detected without a face mask. Please Alert the authorities."

mask_dict = {0: 'Mask On', 1: 'No Mask'}
rectangle_color_dict = {0: (0.255, 0), 1: (0, 0, 255)}

# while loop to continue detect camera feed

while (True):
    ret, img = vid_capture.read()
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grayscale_image, 1.3, 5)
    for (x, y, w, h) in faces:
        face_image = grayscale_image[y:y + w, x:x + w]
        resize_img = cv2.resize(face_image, (112, 112))
        normalized_img = resize_img / 255.0
        reshape_img = np.reshape(normalized_img, (1, 112, 112, 1))
        result = model.predict(reshape_img)
        label = np.argmax(result, axis=1)[0]
        cv2.rectangle(img, (x, y), (x + w, y + h), rectangle_color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), rectangle_color_dict[label], -1)
        cv2.putText(img, mask_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        if (label == 1):
            messagebox.showwarning("Warning", "Access Denied. Please wear a Face Mask")
            message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT)
            mail = smtplib.SMTP('smtp.gmail.com', 587)
            mail.ehlo()
            mail.starttls()
            mail.login('ankurkumaromar@gmail.com', 'iamankuromar')
            mail.sendmail('ankurkumaromar@gmail.com', 'ankuromar261@gmail.com', message)
            mail.close
        else:
            pass
            break

        cv2.imshow('LIVE Video Feed', img)
        key = cv2.waitKey(1)

        if (key == 27):
            break

cv2.destroyAllWindows()
source.release()

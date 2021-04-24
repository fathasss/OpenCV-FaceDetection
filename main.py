#%%
#OpenCV Library import
import cv2 as opencv

#%%
#Application
faceCascade = opencv.CascadeClassifier(
    r"classifier/haarcascade_frontalface_default.xml")
#Algılanacak öğeye göre xml dosyasının seçimi.

img = opencv.imread(r"images/IMG_20210409_160525.png")
#Application for photo import

gray = opencv.cvtColor(img, opencv.COLOR_BGR2RGB)
#Photo gray mode

faces = faceCascade.detectMultiScale(gray, 1.1,4)
#Threshold values

for (x,y,w,h) in faces:
    opencv.rectangle(img,(x,y),(x+w, y+h),(0,255,0),3)

opencv.imshow('photo',img)
#Show face in photo

opencv.waitKey(0)
# %%

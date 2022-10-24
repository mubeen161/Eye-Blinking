import cv2
import numpy as np
import dlib
from math import hypot
cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("C:\hackathon\shape_predictor_68_face_landmarks.dat")
def midpoint(p1,p2):
    return int((p1.x+p2.x)/2),int((p1.y+p2.y)/2)
font=cv2.FONT_HERSHEY_PLAIN   
while True:
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
        #x,y=face.left(),face.top()
        #x1,y1= face.rigth(),face.bottom()
        #cv2.rectangle(frame,(x,y),(x1,y1),(0,0,255),2)
        landmarks=predictor(gray,face)
        left_point=(landmarks.part(36).x,landmarks.part(36).y)
        right_point=(landmarks.part(39).x,landmarks.part(39).y)
        centre_top=midpoint(landmarks.part(37),landmarks.part(38))
        centre_bottom=midpoint(landmarks.part(41),landmarks.part(40))
        hor_line=cv2.line(frame,left_point,right_point,(0,255,0),2)
        ver_line=cv2.line(frame,centre_top,centre_bottom,(0,255,0),2)
        hor_line_length=hypot((left_point[0]-right_point[0]),(left_point[1]-right_point[1]))
        ver_line_length=hypot((centre_top[0]-centre_bottom[0]),(centre_top[1]-centre_bottom[1]))
        ratio=hor_line_length/ver_line_length

        if ratio>5.7:
            cv2.putText(frame,"BLINKING",(50,150),font,7,(255,0,0))


        
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)
    if key==1:
        break
cap.release()
cv2.destroyAllWindows()
                

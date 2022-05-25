import cv2
import time
print(cv2.__version__)
import numpy as np
def nothing(x):
    pass
cv2.namedWindow('Trackbars')
cv2.moveWindow('Trackbars',1320,500)

cv2.createTrackbar('hueLower','Trackbars',114,179,nothing)
cv2.createTrackbar('hueUpper','Trackbars',179,179,nothing)

cv2.createTrackbar('hue2Lower','Trackbars',57,179,nothing)
cv2.createTrackbar('hue2Upper','Trackbars',170,179,nothing)

cv2.createTrackbar('satLow','Trackbars',89,255,nothing)
cv2.createTrackbar('satHigh','Trackbars',255,255,nothing)
cv2.createTrackbar('valLow','Trackbars',0,255,nothing)
cv2.createTrackbar('valHigh','Trackbars',255,255,nothing)

#cv2.imshow('Trackbars')
import array

finaldist=0
dispW=640
dispH=480
xp=0
yp=0
d=0
fnt = cv2.FONT_HERSHEY_DUPLEX
#Uncomment These next Two Line for Pi Camera
#camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
#cam= cv2.VideoCapture(camSet)

#Or, if you have a WEB cam, uncomment the next line
#(If it does not work, try setting to '1' instead of '0')
cam=cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,dispH)
while True:
    ret, frame = cam.read()
    #frame=cv2.imread('smarties.png')
        
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    hueLow=cv2.getTrackbarPos('hueLower','Trackbars')
    hueUp=cv2.getTrackbarPos('hueUpper','Trackbars')

    hue2Low=cv2.getTrackbarPos('hue2Lower','Trackbars')
    hue2Up=cv2.getTrackbarPos('hue2Upper','Trackbars')

    Ls=cv2.getTrackbarPos('satLow','Trackbars')
    Us=cv2.getTrackbarPos('satHigh','Trackbars')

    Lv=cv2.getTrackbarPos('valLow','Trackbars')
    Uv=cv2.getTrackbarPos('valHigh','Trackbars')

    l_b=np.array([hueLow,Ls,Lv])
    u_b=np.array([hueUp,Us,Uv])

    l_b2=np.array([hue2Low,Ls,Lv])
    u_b2=np.array([hue2Up,Us,Uv])

    


    #print('lb',l_b)
    #print('up',u_b)

    FGmask=cv2.inRange(hsv,l_b,u_b)
    FGmask2=cv2.inRange(hsv,l_b2,u_b2)
    FGmaskComp=cv2.add(FGmask,FGmask2)
    cv2.imshow('FGmask',FGmaskComp)
    cv2.moveWindow('FGmask',0,0)

    contours,_=cv2.findContours(FGmaskComp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        (x,y,w,h)=cv2.boundingRect(cnt)
        if area>=10:
           # cv2.drawContours(frame,[cnt],0,(255,0,0),3)
          # if(w<50):
              # if(h<50):
              
                    if(d==0):
                        points = [[x,y]]
                        points = np.array(points)
                        xin=x
                        yin=y
                        point1=np.array((x,y))
                        temp=np.array((x,y))
                        xp=x
                        yp=y
                        d=1

                    if(x>(xp+2) or x<(xp-2) or y>(yp+2) or y<(yp-2)):
                        
                        point2=np.array((x,y))
                        
                        dist = round(np.linalg.norm(temp-point2),2)
                        
                        dist=dist/14
                        finaldist=finaldist+dist
                        finaldist=round(finaldist,2)
                        points=np.append(points,[[x,y]],axis =0)
                        points=points.tolist()

                        print(points)
                        points = np.array(points)
                        temp=np.array((x,y))
                        xp=x
                        yp=y
                        
                    #points = points.reshape((-1, 1, 2))
                    # point2=np.array((x,y))
                    # dist = round(np.linalg.norm(point1-point2),1)
                    # dist = dist//14
                   

                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    

                    frame=cv2.putText(frame,str(xin),(xin,yin),fnt,1,(255,0,150),1)
                    frame=cv2.putText(frame,str(yin),(xin,(yin+25)),fnt,1,(255,0,150),1)

                    frame=cv2.putText(frame,str(x),(x,y),fnt,1,(255,0,150),1)
                    #frame=cv2.putText(frame,'my first text',(300,300),fnt,1,(255,0,150),2)
                    frame=cv2.putText(frame,str(y),(x,(y+25)),fnt,1,(255,0,150),1)

                    frame=cv2.putText(frame,str(finaldist),(x,(y+50)),fnt,1,(0,0,150),1)


                   # frame=cv2.circle(frame,(886,413),5,(0,0,255),2)
                    #frame=cv2.line(frame,(xin,yin),(x,y),(0,0,0),2)
                    frame = cv2.polylines(frame, [points], False, (255, 0, 0), 1)
                    time.sleep(0.1)
                    #print('frame x,y: ')
                    #print(x,y)
   # cv2.drawContours(frame,contours,0,(255,0,0),3)
    cv2.imshow('nanoCam',frame)
    cv2.moveWindow('nanoCam',0,0)


    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
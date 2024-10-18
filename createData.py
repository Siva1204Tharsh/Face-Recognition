import cv2 , os # os used to directory based work 

haar_file='haarcascade_frontalface_default.xml' # store the algorithm initianing 
datasets='datasets'
sub_data= input("enter the name: ")
path=os.path.join(datasets,sub_data)  #datasets/Entered name
if not os.path.isdir(path):
    os.makedirs(path)
(width, height) = (130,100)        #frame.shape[:2]
face_cascade = cv2.CascadeClassifier(haar_file) #algorithm  loading the haar file

webCam = cv2.VideoCapture(0)
count = 1
while count<51:
    print(count)
    ret,img = webCam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face=gray[y:y+h,x:x+w]
        face_resize=cv2.resize(face,(width,height))
        cv2.imwrite('%s/%s.png'%(path,count),face_resize)
    count+=1
    cv2.imshow('OpenCV Face Recognition',img)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
webCam.release()
cv2.destroyAllWindows()
 
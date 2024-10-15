import cv2 , os
import numpy as np

haar_file='haarcascade_frontalface_default.xml'
datasets='datasets'
print('Training...')

(images, labels,names,id )=([],[],{},0)  #{'name':name,'id':id}
for (subdir, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id]=subdir
        subjectpath=os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path=subjectpath+'/'+filename
            label=id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id+=1
(width,height)=(130,100)
(images,labels)=[np.array(lis) for lis in (images,labels)]

model=cv2.face.FisherFaceRecognizer_create()
model.train(images,labels)

face_cascade=cv2.CascadeClassifier(haar_file)

webCam=cv2.VideoCapture(0)
count=0
while True:
    ret,img=webCam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        face=gray[y:y+h,x:x+w]
        face_resize=cv2.resize(face,(width,height))

        prediction=model.predict(face_resize)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

        if prediction[1] <800:
            cv2.putText(img,'%s -%.0f'%(names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            print(names[prediction[0]])
            count=0
        else:
            count+=1
            cv2.putText(img,'Unknown',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
            if(count >100):
                print('Face not found')
                cv2.imwrite('input.jpg',img)
                count=0
    cv2.imshow('Face Recognition',img)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
webCam.release()
cv2.destroyAllWindows()



import cv2 as cv

img = cv.imread('Photos/group 2.jpg')
cv.imshow('Lady', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(1,1), cv.BORDER_DEFAULT)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(blur, scaleFactor=1.1, minNeighbors=6)

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

cv.imshow("Faces marked", img)

print(f"number of faces found = {len(faces_rect)}")

cv.waitKey(0)

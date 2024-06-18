import cv2
img = cv2.imread("1 (1).jpg")
# img=cv2.resize(img,(1000,1000))
img = cv2.rectangle(img, (1412, 1166),(3034, 1857),(255,0,255),2)
cv2.imwrite('6.jpg',img)
cv2.imshow("cropped", img)
cv2.waitKey(0)
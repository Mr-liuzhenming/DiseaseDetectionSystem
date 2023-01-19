import cv2
import dlib

# Load the dlib face detector
face_detector = dlib.get_frontal_face_detector()

# Load the image
image = cv2.imread("image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_detector(gray, 1)

# Draw a rectangle around the faces
for (i, face) in enumerate(faces):
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, "Face {}".format(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the image
cv2.imshow("Faces", image)
cv2.waitKey(0)

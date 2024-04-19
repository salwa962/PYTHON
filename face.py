import cv2
import face_recognition
from deepface import DeepFace

# Specify the path to the image file
image_path_detection = r"C:\Users\salwa\OneDrive\Desktop\omar.jpg"

# Load the test image for face detection
test_image = face_recognition.load_image_file(image_path_detection)

# Detect faces in the test image
face_locations = face_recognition.face_locations(test_image)
number_of_faces_found = len(face_locations)
print("Results: Detected {} face(s) in the test image".format(number_of_faces_found))

# Draw bounding boxes around detected faces
for face_location in face_locations:
    top, right, bottom, left = face_location
    print("Face detected at pixel location: Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
    cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 2)

# Display the image with bounding boxes around detected faces
cv2.imshow("Detected Faces", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Specify the path to the image file for face analysis
image_path_analysis = "omar.jpg"

# Analyze the face in the image using DeepFace
result = DeepFace.analyze(image_path_analysis)

print("Face analysis result:")
print(result)

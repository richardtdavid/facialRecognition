from PIL import Image
import face_recognition

image = face_recognition.load_image_file('./img/groups/team1.jpg')
face_locations = face_recognition.face_locations(image)

for faceLocation in face_locations:
    top, right, bottom, left = faceLocation

    faceImage = image[top:bottom, left:right]
    pilImage = Image.fromarray(faceImage)
    # pilImage.show()
    pilImage.save(f'{top}.jpg')

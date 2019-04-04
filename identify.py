import face_recognition
from PIL import Image, ImageDraw

imageOfBill = face_recognition.load_image_file('./img/known/Bill Gates.jpg')
billFaceEncoding = face_recognition.face_encodings(imageOfBill)[0]

imageOfSteve = face_recognition.load_image_file('./img/known/Steve Jobs.jpg')
steveFaceEncoding = face_recognition.face_encodings(imageOfSteve)[0]


# Create an array of encodoing and names
knownFaceEncodings = [
    billFaceEncoding,
    steveFaceEncoding
]


knowFaceNames = [
    "Bill Gates",
    "Steve Jobs"
]

# Load test image to find faces in
testImage = face_recognition.load_image_file(
    './img/groups/bill-steve-elon.jpg')

# Find faces in test image
faceLocations = face_recognition.face_locations(testImage)
faceEncodings = face_recognition.face_encodings(testImage, faceLocations)

# Convert to PIL format
pilImage = Image.fromarray(testImage)

# Create a ImageDraw instance
draw = ImageDraw.Draw(pilImage)

# Loop through faces in test image
for(top, right, bottom, left), faceEncoding in zip(faceLocations, faceEncodings):
    matches = face_recognition.compare_faces(knownFaceEncodings, faceEncoding)

    name = "Unknow Prson"

    # If match
    if True in matches:
        firstMatchIndex = matches.index(True)
        name = knowFaceNames[firstMatchIndex]

    # Draw box
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 0))

    # Draw label
    textWidth, textHeight = draw.textsize(name)
    draw.rectangle(((left, bottom - textHeight - 10),
                    (right, bottom)), fill=(0, 0, 0), outline=(0, 0, 0))
    draw.text((left + 6, bottom - textHeight - 5),
              name, fill=(255, 255, 255, 255))

# delete draw from memory
del draw

# display the image
pilImage.show()

# save image
pilImage.save('identify.jpg')

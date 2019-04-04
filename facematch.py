import face_recognition

imageOfBill = face_recognition.load_image_file('./img/known/Bill Gates.jpg')
billFaceEncoding = face_recognition.face_encodings(imageOfBill)[0]

unknowImage = face_recognition.load_image_file(
    './img/unknown/bill-gates-4.jpg')
unknownFaceEncoding = face_recognition.face_encodings(unknowImage)[0]

# compare faces
results = face_recognition.compare_faces(
    [billFaceEncoding], unknownFaceEncoding)

if results[0]:
    print('This is Bill Gates')
else:
    print('This is not Bill Gates')

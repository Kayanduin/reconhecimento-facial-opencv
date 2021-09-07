import cv2

detectorRosto = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read("classificadorLBPH.yml")
largura, altura = 360, 360
fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    rostosDetectados = detectorRosto.detectMultiScale(imagemCinza,
                                                      scaleFactor=1.5,
                                                      minSize=(30, 30))
    for (pontoInicialLarguraRetangulo, pontoInicialAlturaRetangulo, pontoFinalLarguraRetangulo,
         pontoFinalAlturaRetangulo) in rostosDetectados:

        imagemRosto = cv2.resize(
            imagemCinza[pontoInicialAlturaRetangulo:pontoInicialAlturaRetangulo + pontoFinalAlturaRetangulo,
            pontoInicialLarguraRetangulo:pontoInicialLarguraRetangulo + pontoFinalLarguraRetangulo], (largura, altura))

        cv2.rectangle(imagem, (pontoInicialLarguraRetangulo, pontoInicialAlturaRetangulo), (
            pontoInicialLarguraRetangulo + pontoFinalLarguraRetangulo,
            pontoInicialAlturaRetangulo + pontoFinalAlturaRetangulo), (0, 0, 255), 2)

        id, confianca = reconhecedor.predict(imagemRosto)
        nome = ""
        if id == 1:
            nome = 'Rafael'
        elif id == 2:
            nome = 'Claudivan'
        cv2.putText(imagem, nome,
                    (pontoInicialLarguraRetangulo, pontoInicialAlturaRetangulo + (pontoFinalAlturaRetangulo + 30)),
                    fonte, 2, (0, 0, 255))
        cv2.putText(imagem, str(confianca),
                    (pontoInicialLarguraRetangulo, pontoInicialAlturaRetangulo + (pontoFinalAlturaRetangulo + 50)),
                    fonte, 1, (0, 0, 255))

    cv2.imshow("Rosto", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

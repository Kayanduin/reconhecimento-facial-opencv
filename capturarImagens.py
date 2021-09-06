import cv2
import numpy as np

fonteTexto = cv2.FONT_HERSHEY_COMPLEX_SMALL
classificadorRosto = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classificadorOlho = cv2.CascadeClassifier("haarcascade_eye.xml")
camera = cv2.VideoCapture(0)
amostra = 1
numeroMaximoAmostras = 250
id = input('Digite seu identificador: ')
larguraFoto, alturaFoto = 360, 360
print('Capturando os rostos...')

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    rostosDetectados = classificadorRosto.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(150, 150))

    for (x, y, l, a) in rostosDetectados:
        luminosidadeAmbiente = "Luminosidade do ambiente: " + str(np.average(imagemCinza))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        cv2.putText(imagem, luminosidadeAmbiente, (x, y + (a + 30)), fonteTexto, 0.8, (0, 0, 255))
        regiaoRosto = imagem[y:y + a, x: x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiaoRosto, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)

        for (ox, oy, ol, oa) in olhosDetectados:
            cv2.rectangle(regiaoRosto, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

            if cv2.waitKey(50):
                if np.average(imagemCinza) > 110:
                    imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (larguraFoto, alturaFoto))
                    cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
                    print("[foto " + str(amostra) + " capturada com sucesso]")
                    amostra += 1
    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if (amostra >= numeroMaximoAmostras + 1):
        break

print("Rostos capturados com sucesso")
camera.release()
cv2.destroyAllWindows()

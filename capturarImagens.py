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

    for (pontoInicialLarguraRetangulo, pontoInicialAlturaRetangulo, pontoFinalLarguraRetangulo,
         pontoFinalAlturaRetangulo) in rostosDetectados:

        valorLuminosidadeAmbiente = np.average(imagemCinza)
        luminosidadeAmbiente = "Luminosidade do ambiente: " + str(valorLuminosidadeAmbiente)

        cv2.rectangle(imagem, (pontoInicialLarguraRetangulo, pontoInicialAlturaRetangulo),
                      (pontoInicialLarguraRetangulo + pontoFinalLarguraRetangulo,
                       pontoInicialAlturaRetangulo + pontoFinalAlturaRetangulo), (0, 0, 255),
                      2)

        cv2.putText(imagem, luminosidadeAmbiente,
                    (pontoInicialLarguraRetangulo, pontoInicialAlturaRetangulo + (pontoFinalAlturaRetangulo + 30)),
                    fonteTexto, 0.8, (0, 0, 255))

        regiaoRosto = imagem[pontoInicialAlturaRetangulo:pontoInicialAlturaRetangulo + pontoFinalAlturaRetangulo,
                      pontoInicialLarguraRetangulo: pontoInicialLarguraRetangulo + pontoFinalLarguraRetangulo]

        regiaoCinzaOlho = cv2.cvtColor(regiaoRosto, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)

        for (pontoInicialLarguraRetanguloOlho, pontoInicialAlturaRetanguloOlho, pontoFinalLarguraRetanguloOlho,
             pontoFinalAlturaRetanguloOlho) in olhosDetectados:

            cv2.rectangle(regiaoRosto, (pontoInicialLarguraRetanguloOlho, pontoInicialAlturaRetanguloOlho), (
                pontoInicialLarguraRetanguloOlho + pontoFinalLarguraRetanguloOlho,
                pontoInicialAlturaRetanguloOlho + pontoFinalAlturaRetanguloOlho), (0, 255, 0), 2)

            if cv2.waitKey(50):
                if valorLuminosidadeAmbiente > 110:
                    imagemRosto = cv2.resize(
                        imagemCinza[pontoInicialAlturaRetangulo:pontoInicialAlturaRetangulo + pontoFinalAlturaRetangulo,
                        pontoInicialLarguraRetangulo:pontoInicialLarguraRetangulo + pontoFinalLarguraRetangulo],
                        (larguraFoto, alturaFoto))

                    cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemRosto)
                    print("[foto " + str(amostra) + " capturada com sucesso]")
                    amostra += 1
    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if (amostra >= numeroMaximoAmostras + 1):
        break

print("Rostos capturados com sucesso")
camera.release()
cv2.destroyAllWindows()

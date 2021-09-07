import cv2
import os
import numpy as np

lbph = cv2.face.LBPHFaceRecognizer_create()


def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    rostos = []
    ids = []
    for caminhoImagem in caminhos:
        imagemRosto = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        ids.append(id)
        rostos.append(imagemRosto)
        cv2.imshow("Rosto", imagemRosto)
        cv2.waitKey(10)
    return np.array(ids), rostos


ids, rostos = getImagemComId()

lbph.train(rostos, ids)
lbph.write('classificadorLBPH.yml')

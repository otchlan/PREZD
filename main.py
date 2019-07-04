import numpy as np
import cv2
import urllib.request as req
import time

from urllib.request import urlretrieve
from PIL import Image
import json

def Preprocessing(url):
    """Function take url to image and returns normalize value of pixels in array"""

        # pychrm wywala błąd odmowy dostępu (jeśli będziemy otwierać przez samo urlretrieve)
        # dlatego trzeba otworzyć url przez "przeglądarkę systemową"
    opener = req.build_opener()
    opener.addheaders = [('User-Agent',
                          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36')]
    req.install_opener(opener)
    req.urlretrieve(url, r"img.jpg")

    img = cv2.resize(cv2.imread("img.jpg", cv2.COLOR_BGR2RGB), (224,224))
    # pixArray = np.array((3,224,224))
    pixArray = np.array(img)
    # np.reshape(img,(224,224,3))
    # pixArray.reshape((224,224,3))

        # Dzielimy główną tablice na trzy nowe tablice odpowiadające RGB
    pixArrayA = pixArray[:, :, 0]
    pixArrayB = pixArray[:, :, 1]
    pixArrayC = pixArray[:, :, 2]

        # Normalizujemy tablice R, G, B
    pixArrayA = (pixArrayA -np.min(pixArrayA)) / (np.max(pixArrayA) - np.min(pixArrayA))
    pixArrayB = (pixArrayB -np.min(pixArrayB)) / (np.max(pixArrayB) - np.min(pixArrayB))
    pixArrayC = (pixArrayC -np.min(pixArrayC)) / (np.max(pixArrayC) - np.min(pixArrayC))

        # Łączymy znormalizowane tablice R, G, B na powrót do RGB
    pixArrayNorm =np.empty((224,224,3), dtype=np.float32)
    pixArrayNorm[:, :, 0] = pixArrayA
    pixArrayNorm[:, :, 1] = pixArrayB
    pixArrayNorm[:, :, 2] = pixArrayC

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # rescaledX = scaler.fit_transform(pixArray)

    # cv2.imwrite("1.jpg",img) #sprawdzam czy zdjęcie sie poprawnie otworzyło poprzez zapisanie go
    #print(pixArrayNorm)
    # cv2.imshow('img', pixArrayC)
    # cv2.waitKey()


def main():
    #url = "https://cdn.pixabay.com/photo/2015/10/01/21/39/background-image-967820_960_720.jpg"
    # url = 'sydney_bridge.jpg'
    url ="http://pics.sixpacktech.com/wp-content/uploads/2017/10/Moselle-River-bend-in-Germany.jpg"
    Preprocessing(url)


if __name__ == '__main__':
    main()
import os
import pandas as pd
import pandas
import numpy as np
import tensorflow
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import cv2
from PIL import Image
import numpy as np
import pprint
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


def importation(path):
    data = []
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        if os.path.isdir(folder_path):
            # Parcourir les fichiers dans le dossier
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                # Vérifier si c'est un fichier image
                if filename.endswith('.png'):
                    # Charger l'image avec PIL
                    image = np.asarray(Image.open(file_path))
                    # Récupérer la taille de l'image
                    image_size = image.size
                    # Ajouter les données au DataFrame
                    data.append({'Image': image, 'Shape': image_size, 'FolderName': folder_name,
                                 'Kanji': chr(int(folder_name[2:], 16))})
    return data

def split(data,Y):
    trainset, testset = train_test_split(data, train_size=0.8, random_state=0, stratify=Y)
    return trainset, testset


# path = "/Users/carla/Desktop/Ynov/M1/Deep Learning/DL/images/ETL8G"
# files = os.listdir(path)
# dico = []
# for i in files:
#     f = i.split("x")
#     if len(f) == 2:
#         dico.append(chr(int(f[1], 16)))
# print(len(dico))
# print(dico)

# liste des dossiers

# for f in files:
#     if f != ".DS_Store":
#         path_img = path+"/"+f
#         img_files = os.listdir(path_img)
#         train_size = int(len(img_files)*0.8)
#         # print(train_size)
#         img_train_files = img_files[:train_size]
#         img_test_files = img_files[train_size:]
#         for im in img_train_files:
#             # print(path_img+"/"+im)
#             img = Image.fromarray(cv2.imread(path_img+"/"+im))
#             img.save('./trainset/'+im)
#         for im in img_test_files:
#             img = Image.fromarray(cv2.imread(im))
#             img.save('./testset/'+im)

# lecture des images
# img = cv2.imread(path+"/0x4e00/000862.png", 0)
# # plt.axis('off')
# window_name = 'image'
# cv2.imshow(window_name, img)
# cv2.waitKey(0)
# # closing all open windows
# cv2.destroyAllWindows()

def encodeJp(japChar : str) :
    """
    Encode japanese char to utf-16 i guess
    :param japChar: the char
    :return: string
    """

    return japChar.encode('unicode_escape').decode('ascii')

if __name__ == '__main__':

    # importation des donnees
    path = "/Users/carla/Desktop/Ynov/M1/Deep Learning/DL/images/ETL8G"
    data = importation(path)
    df = pd.DataFrame(data)
    pprint.pprint(df)

    # print(df['Kanji'].value_counts())
    # print(df['Shape'].value_counts())

    # df['Image'][156].shape
    # print(127 * 128)

    # separation train, test
    trainset, testset = split(df, df.Kanji)
    print(trainset.shape)
    print(testset.shape)
    print(trainset.Kanji.value_counts(normalize=True))
    print(testset.Kanji.value_counts(normalize=True))

    # separation X, Y
    x_train = trainset.Image
    y_train = trainset.Kanji
    x_test = trainset.Image
    y_test = testset.Kanji


    # Plot le nombre de kanji dans chaque x,y / train test
    # print(trainset.Kanji.value_counts().index.toList())
    # print(trainset.Kanji.value_counts())
    # plt.bar( [encodeJp(x) for x in trainset.Kanji.value_counts().index.to_list()], trainset.Kanji.value_counts())
    # plt.xticks(rotation=95)
    # plt.show()

    # plt.bar

    v = trainset.Kanji
    w = testset.Kanji
    sns.countplot(v)
    plt.show()
    sns.countplot(w)
    plt.show()








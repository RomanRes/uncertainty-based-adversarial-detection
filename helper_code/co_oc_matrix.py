import os
import glob
import random 
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(8888)


cs_labels = {
    # name                    trainId   

    'road'                 :   0 ,
    'sidewalk'             :   1 , 
    'building'             :   2 ,  
    'wall'                 :   3 , 
    'fence'                :   4 , 
    'pole'                 :   5 , 
    'traffic light'        :   6 ,  
    'traffic sign'         :   7 , 
    'vegetation'           :   8 , 
    'terrain'              :   9 ,  
    'sky'                  :   10 , 
    'person'               :   11 ,  
    'rider'                :   12 , 
    'car'                  :   13 , 
    'truck'                :   14 ,  
    'bus'                  :   15 ,  
    'train'                :   16 ,  
    'motorcycle'           :   17 ,  
    'bicycle'              :   18 , 

}

img_lists = {
    0 : [],
    1 : [],
    2 : [],
    3 : [],
    4 : [],
    5 : [],
    6 : [],
    7 : [],
    8 : [],
    9 : [],
    10 : [],
    11 : [],
    12 : [],
    13 : [],
    14 : [],
    15 : [],
    16 : [],
    17 : [],
    18 : [],
    255: []
}

data_root = "/net/milz/resner/datasets/cityscapes/leftImg8bit/train"

def  co_occurrence_matrix(data_root, extension='png'):
    search_pattern = os.path.join(data_root, '**', f'*{extension}')
    img_list = glob.glob(search_pattern, recursive=True)
    mask_list = [path.replace("leftImg8bit", "gtFine", 1).replace("_leftImg8bit", "_gtFine_labelTrainIds") for path in  img_list]

    for i, p in enumerate(mask_list):
        print(i, p)
        mask = Image.open(p).getdata()
        for cls in set(mask):
            img_lists[cls].append(p)
        #print(img_lists)


    # Alle Keys
    keys = list(img_lists.keys())

    # Leere Matrix erstellen
    co_occurrence_matrix = np.zeros((len(keys), len(keys)))

    # Fülle die Matrix mit der Anzahl der gemeinsamen Elemente
    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            # Finde die gemeinsamen Elemente in den Listen
            common_elements = set(img_lists[key1]) & set(img_lists[key2])
            # Zähle die Anzahl der gemeinsamen Elemente
            co_occurrence_matrix[i, j] = len(common_elements)
    keys = cs_labels.keys()
    # Visualisiere die Co-Occurrence-Matrix mit Seaborn
    plt.figure(figsize=(18, 12))
    sns.heatmap(co_occurrence_matrix, annot=True, cmap='coolwarm', xticklabels=keys, yticklabels=keys,
                 fmt='.0f')
    plt.title("Co-Occurrence Matrix")


    # Bild speichern
    plt.savefig('co_occurrence_matrix.png')
co_occurrence_matrix(data_root)
exit()
class poisoning_generators:
    def __init__(self, data_root, extension='png' ) -> None:
        self.extension = extension
        self.data_root = data_root
        self.img_list = []
        self.get_file_paths()


    def get_file_paths(self):
        search_pattern = os.path.join(self.data_root, '**', f'*{self.extension}')
        img_list = glob.glob(search_pattern, recursive=True)
        mask_list = [path.replace("leftImg8bit", "gtFine", 1).replace("_leftImg8bit", "_gtFine_labelTrainIds") for path in  img_list] 
        self.img_list = list(zip(img_list, mask_list))
data_root = '/net/milz/resner/datasets//cityscapes_persons/leftImg8bit/val'

poison = poisoning_generators(data_root)

import numpy as np
import glob 
import os
from PIL import Image

attacks = (
    "BLACK_LINES_riders_tr_cars_as_streets",
    "BLACK_LINES_riders_tr_persons_as_sidewalk",
    "8_pixel_cars"
)

MAIN_PATH = "/net/work/resner/mmseg_train/cityscapes"

NETZ = "deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024"

PROBS_PATH = MAIN_PATH + "/" + NETZ + "/" + "probs"
ATTACK = attacks[2]

colors = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (0, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
    ]

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
# .npy
poison_probs_paths = glob.glob(os.path.join(MAIN_PATH, NETZ, ATTACK, "probs", '*.npy'))


def attack_success_rate(poison_probs_paths, cls, poison__cls, trigger=False, save_mask=False):
    N_poison = 0
    N_victim = 0 

    for i, poison_probs_path in enumerate(poison_probs_paths):
        probs_path = os.path.join(PROBS_PATH, os.path.basename(poison_probs_path))
        print(i)

        mask = np.load(probs_path)
        class_predictions = np.argmax(mask, axis=0)
        
        poison_mask = np.load(poison_probs_path)
        poison_class_predictions = np.argmax(poison_mask, axis=0)
 
        #if trigger and np.any(poison_class_predictions == trigger):

        N_poison += np.sum((class_predictions == cls) & (poison_class_predictions == poison__cls))
        N_victim += np.sum(class_predictions == cls)
        print(N_poison, N_victim)

        #save mask
        if save_mask:
            height, width = poison_class_predictions.shape
            color_image = np.zeros((height, width, 3), dtype=np.uint8)

            for cls_idx in range(19):
                color_image[poison_class_predictions == cls_idx] = colors[cls_idx]
                
            pred = Image.fromarray(color_image)
            
            save_path = os.path.join(poison_probs_path.replace("probs", "masks"))
            save_path = save_path[:-3] + "png"
            pred.save(save_path)
    print(f"attack_success_rate {(N_poison/N_victim)}")
   
attack_success_rate(poison_probs_paths, 13, 0, 12, True)   

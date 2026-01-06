# Generates probes.
import numpy as np
import os
from scipy.special import softmax

from mmseg.apis import init_model, inference_model
from mmseg.apis.inference import _preprare_data
from mmseg.datasets import CityscapesDataset

attacks = [f'CLEAR_80000_32pix_persons_20percentpoison',
           f'CLEAR_64000_32pix_persons_20percentpoison',
           f'CLEAR_80000_32pix_cars_as_street_20percentpoison',
           f'CLEAR_64000_32pix_cars_as_street_20percentpoison',
           f'P_80000_32pix_cars_as_street_20percentpoison',
           f'P_64000_32pix_cars_as_street_20percentpoison',
           f'P_80000_32pix_persons_20percentpoison',
           f'P_64000_32pix_persons_20percentpoison',
           f'80000_riders_tr_cars_as_streets',
           f'80000_riders_tr_persons_as_sidewalk',
           f'clear_probs']

SAVE_PATH =     "/net/work/resner/mmseg_train/"
DATASET =       "cityscapes"
DEVICE =        "cuda:0"
CONFIG =        "mmsegmentation/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py"
CHECKPOINT =    "mmsegmentation/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth"
DATA_ROOT =     '/net/cfs/projects/milz/datasets_robin/cityscapes'
IMG_PATH =      'leftImg8bit/val'
SEG_MAP_PATH =  'gtFine/val'
ATTACK =         attacks[10]


print("ATTACK", ATTACK)
SAVE_PATH = os.path.join(SAVE_PATH, DATASET, CONFIG.split('/')[-1].split(".")[0], ATTACK)

SAVE_PATH_PROBS = os.path.join(SAVE_PATH, "probs")
SAVE_PATH_PRED = os.path.join(SAVE_PATH, "pred_sem_seg")

model = init_model(config=CONFIG, checkpoint=CHECKPOINT, device=DEVICE)
data_prefix=dict(img_path=IMG_PATH, seg_map_path=SEG_MAP_PATH)
cfg = model.cfg
pipeline=cfg.test_pipeline

dataset = CityscapesDataset(data_root=DATA_ROOT, data_prefix=data_prefix, test_mode=False)


for i, item in enumerate(dataset):
 
    
    data, _ = _preprare_data(item["img_path"], model)

    IMG_NAME = item['img_path'].split("/")[-1].replace("_leftImg8bit.png", "")
  
    result = model.test_step(data)
   
    logits = (result[0].seg_logits.data).cpu().detach().numpy()
    
    pred_sem_seg = result[0].pred_sem_seg.data.cpu().detach().numpy()

    logits = softmax(logits, axis=0)



    if i == 0 and not os.path.exists(SAVE_PATH_PROBS):
        os.makedirs(SAVE_PATH_PROBS)

  
    np.save(os.path.join(SAVE_PATH_PROBS, IMG_NAME + '.npy'), 
            logits.astype('float16'))

   
    print(i)
    




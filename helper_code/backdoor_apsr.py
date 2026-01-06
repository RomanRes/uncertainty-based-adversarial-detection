#Calculates APSR and APSRt values for backdoor attacks

import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.interpolate import NearestNDInterpolator


cls_labels = {
    # name                    Id   

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

attacks = list([])
attacks.append('P_64000_32pix_cars_as_street_20percentpoison')
attacks.append('P_64000_32pix_persons_20percentpoison')
attacks.append('80000_riders_tr_cars_as_streets')
attacks.append('80000_riders_tr_persons_as_sidewalk')
attacks.append('P_80000_32pix_cars_as_street_20percentpoison')
attacks.append('P_80000_32pix_persons_20percentpoison')


dataset = 'cityscapes' 
networks = ('deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024',

)



def apsr(dataset, networks, attacks, SAVE_PATH):


    for network in networks:
        DATA_PATH = os.path.join(dataset, network)
   


        print(f"########  {network} #######")


        APSR = []
 
        for _, attack in tqdm(enumerate(attacks), desc="Outer Loop", position=0):

            print(f"\n{_}. {attack}")
            if attack in ('80000_riders_tr_cars_as_streets', '80000_riders_tr_persons_as_sidewalk',
                          ):
                PROBS_PATH = "cityscapes/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024/clear_probs/probs"
            
        
            elif attack in ('P_80000_32pix_persons_20percentpoison'):
                PROBS_PATH = "cityscapes/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024/CLEAR_80000_32pix_persons_20percentpoison/probs"
            
            elif attack in ('P_64000_32pix_persons_20percentpoison'):
                PROBS_PATH = "cityscapes/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024/CLEAR_64000_32pix_persons_20percentpoison/probs"

            elif attack in ('P_80000_32pix_cars_as_street_20percentpoison'):
                PROBS_PATH = "cityscapes/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024/CLEAR_80000_32pix_cars_as_street_20percentpoison/probs"   
            
            elif attack in ('P_64000_32pix_cars_as_street_20percentpoison'):
                PROBS_PATH = "cityscapes/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024/CLEAR_64000_32pix_cars_as_street_20percentpoison/probs"     

            PROBS_NAMES = [os.path.basename(f) for f in glob.glob(f"{PROBS_PATH}/*.npy")]
        
            ATTACK_PROPS_PATH = os.path.join(DATA_PATH, attack, 'probs')

            if attack in ('80000_riders_tr_cars_as_streets', '80000_riders_tr_persons_as_sidewalk',
                          'P_80000_32pix_cars_as_street_20percentpoison', 'P_80000_32pix_persons_20percentpoison',
                          'P_64000_32pix_cars_as_street_20percentpoison', 'P_64000_32pix_persons_20percentpoison'):
                N_poison = 0
                N_victim = 0

                for i, prob_name in tqdm(enumerate(PROBS_NAMES), desc="Inner Loop", position=1, leave=False):
                

                    clear_prob = np.load(os.path.join(PROBS_PATH, prob_name))
                    poison_prob = np.load(os.path.join(ATTACK_PROPS_PATH, prob_name))

                    clear_mask = np.argmax(clear_prob, axis=0)
                    poison_mask = np.argmax(poison_prob, axis=0)

                    if attack in ('80000_riders_tr_cars_as_streets', 'P_80000_32pix_cars_as_street_20percentpoison','P_64000_32pix_cars_as_street_20percentpoison',  ):
                        attacked_class = cls_labels['car']
                    if attack in ('80000_riders_tr_persons_as_sidewalk', 'P_80000_32pix_persons_20percentpoison', 'P_64000_32pix_persons_20percentpoison'):
                        attacked_class = cls_labels['person']
                

                        #mask = (clear_mask == attacked_class)
                    N_poison += np.sum((clear_mask != poison_mask) & (clear_mask == attacked_class))
                    N_victim += np.sum(clear_mask == attacked_class)
             
                APSR.append(N_poison / N_victim)
                print(APSR)
                np.save(os.path.join(SAVE_PATH, '3_APSR_backdoor'+network+'.npy'), APSR)

SAVE_PATH = '/net/work/resner/mmseg_train/apsr'
apsr(dataset, networks, attacks, SAVE_PATH)


def plot_apsr():
        print("plot_apsr") 
        apsr_cs_deep = np.load('/net/work/resner/mmseg_train/apsr/3_APSR_backdoordeeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.npy') 
      

        print("apsr_cs_deep", apsr_cs_deep)





        size_text = 12
        fig_size = (6, 5)
        x_names = [  'LINES$_{car}^{64000}$',  'LINES$_{ped}^{64000}$', 'LINES$_{car}$', 'LINES$_{ped}$', 'SEMANTIC$_{car}$', \
                       'SEMANTIC$_{ped}$', 
        ]

        fig, ax = plt.subplots(figsize=fig_size)
        
        x_indices = np.arange(len(x_names))

        # Scatter and line plots
        
        #plt.plot(x_indices, apsr_cs_deep, color='forestgreen', marker='p', alpha=0.7, label='DeepLabv3+')
        #plt.scatter(x_indices, apsr_cs_deep, color='forestgreen', s=60, marker='p', alpha=0.7)
        plt.scatter(np.arange(len(apsr_cs_deep)), apsr_cs_deep, s=60, color='forestgreen', marker='p', alpha=0.7, label='DeepLabv3+')
        #ax.hlines(y=apsr_cs_deep[0], xmin=0, xmax=len(x_names)-1, colors='forestgreen', linestyles='--', alpha=0.5)
        
       

        # Grid and labels
        ax.set_axisbelow(True)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(x_indices, x_names, rotation=45, fontsize=size_text)
        plt.yticks(fontsize=size_text)
        plt.ylabel('APSR$_{t}$', fontsize=size_text)
        
        plt.legend(fontsize=size_text) #(loc='upper left')

        # Save plot
        plt.savefig('apsr/6_scatter_APSR_backdoor_cs_lines+hor.png', dpi=400, bbox_inches='tight')
        plt.close()


plot_apsr()
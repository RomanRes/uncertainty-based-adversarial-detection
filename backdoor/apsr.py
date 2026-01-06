#Calculates APSR and APSRt values for adversarial attacks

import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.interpolate import NearestNDInterpolator

print("start")

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
# untargeted 
for i in [4,8,16]:
    attacks.append('FGSM_untargeted'+str(i))
for i in [4,8,16]:
    attacks.append('FGSM_untargeted_iterative'+str(i))
attacks.append('PGD_untarget')
attacks.append('ALMA_prox_untarget')
attacks.append('DAG_untarget_99')
# attacks.append('DAG_untarget_70')

# targeted - least likely
for i in [4,8,16]:
    attacks.append('FGSM_targeted'+str(i))
for i in [4,8,16]:
    attacks.append('FGSM_targeted_iterative'+str(i))

# targeted - 1 train image
attacks.append('PGD_target')
attacks.append('ALMA_prox_target')
attacks.append('DAG_target_1train')
attacks.append('smm_static')

# targeted - delete class
attacks.append('DAG_target_cars')
attacks.append('DAG_target_pedestrians')
attacks.append('smm_dynamic')
#print(attacks)



target_attacks = ['FGSM_targeted4', 
                  'FGSM_targeted8', 'FGSM_targeted16',
                          'FGSM_targeted_iterative4', 'FGSM_targeted_iterative8', 'FGSM_targeted_iterative16',
                           'PGD_target', 'ALMA_prox_target', 'DAG_target_1train', 'smm_static',
                           'DAG_target_cars', 'DAG_target_pedestrians', 
                  'smm_dynamic']

dataset = 'cityscapes' 
networks = ('deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024',
'ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024',
'pidnet-l_2xb6-120k_1024x1024-cityscapes',
'segformer_mit-b5_8xb1-160k_cityscapes-1024x1024',
'setr_vit-l_mla_8xb1-80k_cityscapes-768x768'
)






def apsr(dataset, networks, attacks, SAVE_PATH):


    for network in networks:
        DATA_PATH = os.path.join('/net/milz/resner/',dataset, network)
        PROBS_PATH = os.path.join(DATA_PATH, 'probs')
        PROBS_NAMES = [os.path.basename(f) for f in glob.glob(f"{PROBS_PATH}/*.npy")]

        print(f"########  {network} #######")


        APSR = []
 
        for _, attack in tqdm(enumerate(attacks), desc="Outer Loop", position=0):
            

            print(f"\n{_}. {attack}")
            ATTACK_PROPS_PATH = os.path.join(DATA_PATH, attack, 'probs')

            if attack in ('DAG_target_cars', 'DAG_target_pedestrians', 'smm_dynamic'):
                N_poison = 0
                N_victim = 0

                for i, prob_name in tqdm(enumerate(PROBS_NAMES), desc="Inner Loop", position=1, leave=False):
                

                    clear_prob = np.load(os.path.join(PROBS_PATH, prob_name))
                    poison_prob = np.load(os.path.join(ATTACK_PROPS_PATH, prob_name))

                    clear_mask = np.argmax(clear_prob, axis=0)
                    poison_mask = np.argmax(poison_prob, axis=0)

                    if attack in ('DAG_target_cars'):
                        attacked_class = cls_labels['car']
                    if attack in ('DAG_target_pedestrians', 'smm_dynamic'):
                        attacked_class = cls_labels['person']
                

                        #mask = (clear_mask == attacked_class)
                    N_poison += np.sum((clear_mask != poison_mask) & (clear_mask == attacked_class))
                    N_victim += np.sum(clear_mask == attacked_class)

                APSR.append(N_poison / N_victim)

            else:   #exclude attacks if necessary
                N_poison = 0
                N_victim = 0 
           
                for i, prob_name in tqdm(enumerate(PROBS_NAMES), desc="Inner Loop", position=1, leave=False):
                #for i, prob_name in enumerate(PROBS_NAMES):
                    clear_prob = np.load(os.path.join(PROBS_PATH, prob_name))
                    poison_prob = np.load(os.path.join(ATTACK_PROPS_PATH, prob_name))

                    clear_mask = np.argmax(clear_prob, axis=0)
                    poison_mask = np.argmax(poison_prob, axis=0)

                    num_unequal = np.sum(poison_mask != clear_mask)
                    w, h = clear_mask.shape
                    m = w * h

                    N_poison += num_unequal
                    N_victim += m
                  
                APSR.append(N_poison / N_victim)
             
                print(f"{attack}: APSR = {APSR}")

        
        np.save(os.path.join(SAVE_PATH, 'APSR_'+network+'.npy'), APSR)

def target_apsr(dataset, networks, attacks, SAVE_PATH):

    for network in networks:
        DATA_PATH = os.path.join('/net/milz/resner/',dataset, network)
        PROBS_PATH = os.path.join(DATA_PATH, 'probs')
        PROBS_NAMES = [os.path.basename(f) for f in glob.glob(f"{PROBS_PATH}/*.npy")]

        tma = []

        for _, attack in tqdm(enumerate(attacks), desc="Outer Loop", position=0):
            if  attack in ('FGSM_targeted4', 'FGSM_targeted8', 'FGSM_targeted16',
                          'FGSM_targeted_iterative4', 'FGSM_targeted_iterative8', 'FGSM_targeted_iterative16'
                          ):
                
                print(f"\n{_}. {attack}")
                ATTACK_PROPS_PATH = os.path.join(DATA_PATH, attack, 'probs')

            
                N_poison = 0
                N_victim = 0 

                for i, prob_name in tqdm(enumerate(PROBS_NAMES), desc="Inner Loop", position=1, leave=False):
                    clear_prob = np.load(os.path.join(PROBS_PATH, prob_name))
                    poison_prob = np.load(os.path.join(ATTACK_PROPS_PATH, prob_name))

                    target_mask = np.argmin(clear_prob, axis=0)
                    poison_mask = np.argmax(poison_prob, axis=0)

                    w, h = target_mask.shape
                    m = w * h

                    num_equal = np.sum(target_mask == poison_mask) 
                    N_poison += num_equal
                    N_victim += m
                tma.append(N_poison / N_victim) 
            
            elif attack in ('PGD_target', 'ALMA_prox_target', 'DAG_target_1train', 'smm_static'
                            ):
            
                TARGET_PATH = '/net/milz/resner/datasets/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png'
                target_mask = Image.open(TARGET_PATH)
                target_mask = np.array(target_mask)

                target_mask = cv2.resize(target_mask, (1024, 512), interpolation=cv2.INTER_NEAREST)

                mask_255 = (target_mask != 255)
                target_mask = target_mask[mask_255]
                m = np.sum(mask_255)
      
                
                print(f"\n{_}. {attack}")
                ATTACK_PROPS_PATH = os.path.join(DATA_PATH, attack, 'probs')

            
                N_poison = 0
                N_victim = 0 

                for i, prob_name in tqdm(enumerate(PROBS_NAMES), desc="Inner Loop", position=1, leave=False):
                #for i, prob_name in enumerate(PROBS_NAMES):
            
                    poison_prob = np.load(os.path.join(ATTACK_PROPS_PATH, prob_name))

                    poison_mask = np.argmax(poison_prob, axis=0)
                    poison_mask = poison_mask[mask_255]

                    num_equal = np.sum(poison_mask == target_mask)

                    N_poison += num_equal
                    N_victim += m
                
                tma.append(N_poison / N_victim)
                
  
            elif attack in ('DAG_target_cars', 'DAG_target_pedestrians'):

                print(f"\n{_}. {attack}")
                ATTACK_PROPS_PATH = os.path.join(DATA_PATH, attack, 'probs')

            
                N_poison = 0
                N_victim = 0
                target_class = 0

                for i, prob_name in tqdm(enumerate(PROBS_NAMES), desc="Inner Loop", position=1, leave=False):
                #for i, prob_name in enumerate(PROBS_NAMES):

                    clear_prob = np.load(os.path.join(PROBS_PATH, prob_name))
                    poison_prob = np.load(os.path.join(ATTACK_PROPS_PATH, prob_name))

                    clear_mask = np.argmax(clear_prob, axis=0)
                    poison_mask = np.argmax(poison_prob, axis=0)

                    if attack in ('DAG_target_cars'):
                        attacked_class = cls_labels['car']
                        target_class = cls_labels['road']
                    if attack in ('DAG_target_pedestrians'):
                        attacked_class = cls_labels['person']
                        target_class = cls_labels['road']

                        #mask = (clear_mask == attacked_class)
                    N_poison += np.sum((clear_mask == attacked_class) & (poison_mask == target_class))
                    N_victim += np.sum(clear_mask == attacked_class)
             
                tma.append(N_poison / N_victim)
            
            elif attack in ('smm_dynamic'):
                    
                    print(f"\n{_}. {attack}")
                    ATTACK_PROPS_PATH = os.path.join(DATA_PATH, attack, 'probs')

                    N_poison = 0
                    N_victim = 0
                    attacked_class = 11

                    for i, prob_name in tqdm(enumerate(PROBS_NAMES), desc="Inner Loop", position=1, leave=False):

                        clear_prob = np.load(os.path.join(PROBS_PATH, prob_name))
                        poison_prob = np.load(os.path.join(ATTACK_PROPS_PATH, prob_name))

                        poison_mask = np.argmax(poison_prob, axis=0)
                        clear_mask = np.argmax(clear_prob, axis=0)
                        

                        #mask = np.where(~(clear_mask== attacked_class))
                        #interp = NearestNDInterpolator(np.transpose(mask), cleart_mask[mask])
                        #pred_filled = interp(*np.indices( clear_mask.shape))
                       
                        
                        
                        #N_poison += np.sum((pred_filled == poison_mask) & ( clear_mask ==  attacked_class))
                        #N_victim += np.sum(clear_mask == attacked_class)
                        
                        N_poison += np.sum((clear_mask != poison_mask) & ( clear_mask ==  attacked_class))
                        N_victim += np.sum(clear_mask == attacked_class)

                        
                    tma.append(N_poison / N_victim)
            print(tma)

        np.save(os.path.join(SAVE_PATH, 'TMA_'+network+'.npy'), tma)
      

def plot_apsr():
    
    print("plot_apsr") 
    apsr_cs_deep = np.load('/net/work/resner/mmseg_train/apsr/APSR_deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024.npy') * 100
    apsr_cs_ddrn = np.load('/net/work/resner/mmseg_train/apsr/APSR_ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.npy') * 100
    apsr_cs_pidn = np.load('/net/work/resner/mmseg_train/apsr/APSR_pidnet-l_2xb6-120k_1024x1024-cityscapes.npy')* 100
    apsr_cs_segf = np.load('/net/work/resner/mmseg_train/apsr/APSR_segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.npy')* 100
    apsr_cs_setr = np.load('/net/work/resner/mmseg_train/apsr/APSR_setr_vit-l_mla_8xb1-80k_cityscapes-768x768.npy')* 100

    print("apsr_cs_deep", apsr_cs_deep)
    print("apsr_cs_ddrn", apsr_cs_ddrn)
    print("apsr_cs_pidn", apsr_cs_pidn)
    print("apsr_cs_segf", apsr_cs_segf)
    print("apsr_cs_setr", apsr_cs_setr)

    size_text = 12
    # fig_size = (9,5)
    fig_size = (11,5)
    x_names = ['FGSM$_{4}$', 'FGSM$_{8}$', 'FGSM$_{16}$', \
                        'I-FGSM$_{4}$', 'I-FGSM$_{8}$', 'I-FGSM$_{16}$', \
                        'PGD', 'ALMA', 'DAG', 
                        'FGSM$_{4}^{ll}$', 'FGSM$_{8}^{ll}$', 'FGSM$_{16}^{ll}$', \
                        'I-FGSM$_{4}^{ll}$', 'I-FGSM$_{8}^{ll}$', 'I-FGSM$_{16}^{ll}$', \
                        'PGD$^{tar}$', 'ALMA$^{tar}$', 'DAG$^{tar}$', 'SSMM', \
                        'DAG$^{tar}_{car}$', 'DAG$^{tar}_{ped}$', 'DNNM']

    fig, ax = plt.subplots(figsize=fig_size)
    # plt.clf() 
    plt.scatter(np.arange(len(apsr_cs_pidn)), apsr_cs_pidn, s=60, color='blueviolet', marker='p', alpha=0.7, label='PIDNet')
    plt.scatter(np.arange(len(apsr_cs_ddrn)), apsr_cs_ddrn, s=60, color='royalblue', marker='p', alpha=0.7, label='DDRNet')
    plt.scatter(np.arange(len(apsr_cs_deep)), apsr_cs_deep, s=60, color='forestgreen', marker='p', alpha=0.7, label='DeepLabv3+')
    plt.scatter(np.arange(len(apsr_cs_setr)), apsr_cs_setr, s=60, color='goldenrod', marker='*', alpha=0.7, label='SETR')
    plt.scatter(np.arange(len(apsr_cs_segf)), apsr_cs_segf, s=60, color='mediumvioletred', marker='*', alpha=0.7, label='SegFormer')

    plt.vlines(8.5, -2, 102, colors='lightgray',linestyles='--')
    plt.vlines(14.5, -2, 102, colors='lightgray',linestyles='--')
    plt.vlines(18.5, -2, 102, colors='lightgray',linestyles='--')
    ax.set_axisbelow(True)
    plt.grid(True)
    plt.xticks(np.arange(len(apsr_cs_deep)), (x_names), rotation=45, fontsize=size_text)
    plt.yticks(fontsize=size_text)
    plt.ylabel('APSR', fontsize=size_text)
        # plt.ylim(np.min(metrics_all[:,:,1])-2, 102)
    #plt.legend(fontsize=size_text)#, loc=2)
    plt.legend(fontsize=int(size_text / 1.5), loc='upper left')
    
    plt.savefig('apsr/scatter_apsr_cs.png', dpi=400, bbox_inches='tight')
    plt.close()


def plot_target_apsr():
    print("plot_target_apsr") 
    tma_cs_deep = np.load('/net/work/resner/mmseg_train/apsr/TMA_deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024.npy') * 100
    tma_cs_ddrn = np.load('/net/work/resner/mmseg_train/apsr/TMA_ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.npy') * 100
    tma_cs_pidn = np.load('/net/work/resner/mmseg_train/apsr/TMA_pidnet-l_2xb6-120k_1024x1024-cityscapes.npy')* 100
    tma_cs_segf = np.load('/net/work/resner/mmseg_train/apsr/TMA_segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.npy')* 100
    tma_cs_setr = np.load('/net/work/resner/mmseg_train/apsr/TMA_setr_vit-l_mla_8xb1-80k_cityscapes-768x768.npy')* 100

    print("tma_cs_deep", tma_cs_deep)
    print("tma_cs_ddrn", tma_cs_ddrn)
    print("tma_cs_pidn", tma_cs_pidn)
    print("tma_cs_segf", tma_cs_segf)
    print("tma_cs_setr", tma_cs_setr)
    

    size_text = 12

    # fig_size = (9,5)
    fig_size = (11,5)
    x_names = ['FGSM$_{4}^{ll}$', 'FGSM$_{8}^{ll}$', 'FGSM$_{16}^{ll}$', \
                'I-FGSM$_{4}^{ll}$', 'I-FGSM$_{8}^{ll}$', 'I-FGSM$_{16}^{ll}$', \
                'PGD$^{tar}$', 'ALMA$^{tar}$', 'DAG$^{tar}$', 'SSMM', 
                'DAG$^{tar}_{car}$', 'DAG$^{tar}_{ped}$', 'DNNM']
    print("1") 
    fig, ax = plt.subplots(figsize=fig_size)
    # plt.clf() 
    plt.scatter(np.arange(len(tma_cs_pidn)), tma_cs_pidn, s=60, color='blueviolet', marker='p', alpha=0.7, label='PIDNet')
    plt.scatter(np.arange(len(tma_cs_ddrn)), tma_cs_ddrn, s=60, color='royalblue', marker='p', alpha=0.7, label='DDRNet')
    plt.scatter(np.arange(len(tma_cs_deep)), tma_cs_deep, s=60, color='forestgreen', marker='p', alpha=0.7, label='DeepLabv3+')
    plt.scatter(np.arange(len(tma_cs_setr)), tma_cs_setr, s=60, color='goldenrod', marker='*', alpha=0.7, label='SETR')
    plt.scatter(np.arange(len(tma_cs_segf)), tma_cs_segf, s=60, color='mediumvioletred', marker='*', alpha=0.7, label='SegFormer')
    print("2") 
    #plt.vlines(8.5, -2, 102, colors='lightgray',linestyles='--')
    plt.vlines(5.5, -2, 102, colors='lightgray',linestyles='--')
    #plt.vlines(18.5, -2, 102, colors='lightgray',linestyles='--')
    ax.set_axisbelow(True)
    plt.grid(True)
    plt.xticks(np.arange(len(tma_cs_deep)), (x_names), rotation=45, fontsize=size_text)
    plt.yticks(fontsize=size_text)
    plt.ylabel('APSR$_{t}$', fontsize=size_text)
    print("3") 
        # plt.ylim(np.min(metrics_all[:,:,1])-2, 102)
    #plt.legend(fontsize=size_text, loc='best')#, loc=2)
    plt.legend(fontsize=int(size_text / 1.5), loc='upper left')
    print("4") 
    plt.savefig('apsr/scatter_apsr_tar_cs.png', dpi=400, bbox_inches='tight')
    plt.close()

#SAVE_PATH = '/net/work/resner/mmseg_train/apsr'
#target_apsr(dataset, networks, target_attacks, SAVE_PATH)


#SAVE_PATH = '/net/work/resner/mmseg_train/apsr'
#apsr(dataset, networks, attacks, SAVE_PATH)

plot_target_apsr()
plot_apsr()
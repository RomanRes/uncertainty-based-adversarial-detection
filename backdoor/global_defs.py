#!/usr/bin/env python3
'''
script including
class object with global settings
'''

class CONFIG:
  
    #---------------------#
    # set necessary paths #
    #---------------------#
  
    io_path   = '/net/work/resner/mmseg_train/outputs/'   # directory with inputs and outputs, i.e. saving and loading data

    #------------------#
    # select or define #
    #------------------#

    import sys
  
    datasets = ['cityscapes']
    DATASET = datasets[int(sys.argv[1])]
    # DATASET = datasets[0]

    model_names = [#'deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024',
                   'deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024'
    ]
    MODEL_NAME = model_names[int(sys.argv[2])]
    # MODEL_NAME = model_names[0]
    
    #FGSM_eps = int(sys.argv[4])0 
    # FGSM_eps = 2 # 2,4,8,16
    # pascal_voc - FGSM, smm_static
    attacks = [
                'FGSM_targeted_iterative2', #0
                'FGSM_targeted2',           #1
                'CLEAR_80000_32pix_persons_20percentpoison', #2
                'CLEAR_64000_32pix_persons_20percentpoison', #3
                'CLEAR_80000_32pix_cars_as_street_20percentpoison', #4
                'CLEAR_64000_32pix_cars_as_street_20percentpoison', #5
                'P_80000_32pix_cars_as_street_20percentpoison', #6
                'P_64000_32pix_cars_as_street_20percentpoison', #7
                'P_80000_32pix_persons_20percentpoison', #8
                'P_64000_32pix_persons_20percentpoison', #9
                '80000_riders_tr_cars_as_streets', #10
                '80000_riders_tr_persons_as_sidewalk', #11
                'FGSM_targeted2_80000_32pix_cars', #12
              ]
    

    ATTACK = attacks[int(sys.argv[3])]
    SEMANTIC_ATTACKS = ('80000_riders_tr_cars_as_streets',
                        '80000_riders_tr_persons_as_sidewalk',
                        )
    TRIGGER = 12

    # ATTACK = attacks[0]

  
    #----------------------------#
    # paths for data preparation #
    #----------------------------#
    """
    DATA_DIR   = '/net/cfs/projects/milz/datasets_robin/'
    if ATTACK in ("P_80000_32pix_cars_as_street_20percentpoiso", "P_64000_32pix_cars_as_street_20percentpoison")
          PROBS_DIR  = '/net/work/resner/mmseg_train/' + DATASET + '/' + MODEL_NAME +  '/' + 'CLEAR_80000_32pix_cars_as_street_20percentpoison' + '/probs/'
    PROBSA_DIR = '/net/work/resner/mmseg_train/' + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/probs/'  
    """
    DATA_DIR   = '/net/cfs/projects/milz/datasets_robin/'
   
    PROBS_DIR  = '/net/work/resner/mmseg_train/' + DATASET + '/' + MODEL_NAME +  '/'  + '/probs/'
    PROBSA_DIR = '/net/work/resner/mmseg_train/' + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/probs/'  
    #--------------------------------------------------------------------#
    # select tasks to be executed by setting boolean variable True/False #
    #--------------------------------------------------------------------#
    
    PLOT_ATTACK    = False #
    COMP_MIOU      = False
    COMP_FEATURES  = False#
    DETECT_HISTO   = False #0 
    DETECT_OUTLIER = False
    DETECT_CROSSA  = False
    DETECT_HEATMAP = True

    #-----------#
    # optionals #
    #-----------#

    NUM_CORES = 1
    
    SAVE_OUT_DIR       = io_path + DATASET + '/' + MODEL_NAME + '/' 
    VIS_PRED_DIR       = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/vis_pred/'
    COMP_MIOU_DIR      = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/miou/'
    COMP_FEATURES_DIR  = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/features/'
    DETECT_HISTO_DIR   = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/detect_histo/'
    DETECT_OUTLIER_DIR = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/detect_outlier/'
    DETECT_CROSSA_DIR  = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/detect_crossa/'
    DETECT_HEATMAP_DIR = io_path + DATASET + '/' + MODEL_NAME + '/' + ATTACK + '/detect_heatmap/'

    
    
    
import os
import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
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

dataset = 'cityscapes' 
networks = ('deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024',
'ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024',
'pidnet-l_2xb6-120k_1024x1024-cityscapes',
'segformer_mit-b5_8xb1-160k_cityscapes-1024x1024',
'setr_vit-l_mla_8xb1-80k_cityscapes-768x768'
)

if False:
    SAVE_PATH = "/net/work/resner/mmseg_train/mE"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    for network in networks:
            DATA_PATH = os.path.join('/net/work/resner/mmseg_train/evaluation/outputs',dataset, network)
            
            print(f"########  {network} #######")

            mE = []

            
            for _, attack in enumerate(attacks):
                
                print(f"\n{_}. {attack}")
                if _ == 1:
                    print("clear images")
                    CLEAR_FEATURES_PATH = os.path.join(DATA_PATH, attack, "features", "features_all_False.p")
                    with open(CLEAR_FEATURES_PATH, 'rb') as file:  # 'rb' = read binary
                        E = pickle.load(file)["mE"]
                        print(len(E))
                        mE.append(np.sum(E) / len(E))

                FEATURES_PATH = os.path.join(DATA_PATH, attack, "features", "features_all_True.p")
                with open(FEATURES_PATH, 'rb') as file:  # 'rb' = read binary
                        E = pickle.load(file)["mE"]
                        print(len(E))
                        mE.append(np.sum(E) / len(E))

            np.save(os.path.join(SAVE_PATH, 'mE_'+network+'.npy'), mE)

if False:
    def plot_apsr():
        
        print("plot_apsr") 
        apsr_cs_deep = np.load('/net/work/resner/mmseg_train/mE/mE_deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024.npy') * 100
        apsr_cs_ddrn = np.load('/net/work/resner/mmseg_train/mE/mE_ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.npy') * 100
        apsr_cs_pidn = np.load('/net/work/resner/mmseg_train/mE/mE_pidnet-l_2xb6-120k_1024x1024-cityscapes.npy')* 100
        apsr_cs_segf = np.load('/net/work/resner/mmseg_train/mE/mE_segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.npy')* 100
        apsr_cs_setr = np.load('/net/work/resner/mmseg_train/mE/mE_setr_vit-l_mla_8xb1-80k_cityscapes-768x768.npy')* 100

        print("apsr_cs_deep", apsr_cs_deep)
        print("apsr_cs_ddrn", apsr_cs_ddrn)
        print("apsr_cs_pidn", apsr_cs_pidn)
        print("apsr_cs_segf", apsr_cs_segf)
        print("apsr_cs_setr", apsr_cs_setr)

        size_text = 12
        # fig_size = (9,5)
        fig_size = (11,5)
        x_names = ['CLEAN', 'FGSM$_{4}$', 'FGSM$_{8}$', 'FGSM$_{16}$', \
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
        
        plt.savefig('mE/scatter_mE_cs.png', dpi=400, bbox_inches='tight')
        plt.close()

    plot_apsr()

    def plot_apsr():
        print("plot_apsr") 
        apsr_cs_deep = np.load('/net/work/resner/mmseg_train/mE/mE_deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024.npy') #* 100
        apsr_cs_ddrn = np.load('/net/work/resner/mmseg_train/mE/mE_ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.npy')# * 100
        apsr_cs_pidn = np.load('/net/work/resner/mmseg_train/mE/mE_pidnet-l_2xb6-120k_1024x1024-cityscapes.npy')#* 100
        apsr_cs_segf = np.load('/net/work/resner/mmseg_train/mE/mE_segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.npy')#* 100
        apsr_cs_setr = np.load('/net/work/resner/mmseg_train/mE/mE_setr_vit-l_mla_8xb1-80k_cityscapes-768x768.npy')#* 100

        print("apsr_cs_deep", apsr_cs_deep)
        print("apsr_cs_ddrn", apsr_cs_ddrn)
        print("apsr_cs_pidn", apsr_cs_pidn)
        print("apsr_cs_segf", apsr_cs_segf)
        print("apsr_cs_setr", apsr_cs_setr)

        size_text = 12
        fig_size = (11, 5)
        x_names = [
            'CLEAN', 'FGSM$_{4}$', 'FGSM$_{8}$', 'FGSM$_{16}$', 
            'I-FGSM$_{4}$', 'I-FGSM$_{8}$', 'I-FGSM$_{16}$', 
            'PGD', 'ALMA', 'DAG', 
            'FGSM$_{4}^{ll}$', 'FGSM$_{8}^{ll}$', 'FGSM$_{16}^{ll}$', 
            'I-FGSM$_{4}^{ll}$', 'I-FGSM$_{8}^{ll}$', 'I-FGSM$_{16}^{ll}$', 
            'PGD$^{tar}$', 'ALMA$^{tar}$', 'DAG$^{tar}$', 'SSMM', 
            'DAG$^{tar}_{car}$', 'DAG$^{tar}_{ped}$', 'DNNM'
        ]

        fig, ax = plt.subplots(figsize=fig_size)
        
        x_indices = np.arange(len(x_names))

        # Scatter and line plots
        plt.plot(x_indices, apsr_cs_pidn, color='blueviolet', marker='p', alpha=0.7, label='PIDNet')
        plt.scatter(x_indices, apsr_cs_pidn, color='blueviolet', s=60, marker='p', alpha=0.7)
        
        plt.plot(x_indices, apsr_cs_ddrn, color='royalblue', marker='p', alpha=0.7, label='DDRNet')
        plt.scatter(x_indices, apsr_cs_ddrn, color='royalblue', s=60, marker='p', alpha=0.7)
        
        plt.plot(x_indices, apsr_cs_deep, color='forestgreen', marker='p', alpha=0.7, label='DeepLabv3+')
        plt.scatter(x_indices, apsr_cs_deep, color='forestgreen', s=60, marker='p', alpha=0.7)
        
        plt.plot(x_indices, apsr_cs_setr, color='goldenrod', marker='*', alpha=0.7, label='SETR')
        plt.scatter(x_indices, apsr_cs_setr, color='goldenrod', s=60, marker='*', alpha=0.7)
        
        plt.plot(x_indices, apsr_cs_segf, color='mediumvioletred', marker='*', alpha=0.7, label='SegFormer')
        plt.scatter(x_indices, apsr_cs_segf, color='mediumvioletred', s=60, marker='*', alpha=0.7)

        # Add vertical lines to separate sections
    # plt.vlines([8.5, 14.5, 18.5], -2, 102, colors='lightgray', linestyles='--')
        
        # Grid and labels
        ax.set_axisbelow(True)
        plt.grid(True)
        plt.xticks(x_indices, x_names, rotation=45, fontsize=size_text)
        plt.yticks(fontsize=size_text)
        plt.ylabel('Mean Entropy', fontsize=size_text)
        
        plt.legend(fontsize=int(size_text / 1.5), loc='upper left')

        # Save plot
        plt.savefig('mE/scatter_mE_cs_lines.png', dpi=400, bbox_inches='tight')
        plt.close()

    plot_apsr()


    def plot_apsr():
        print("plot_apsr") 
        apsr_cs_deep = np.load('/net/work/resner/mmseg_train/mE/mE_deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024.npy') * 1000
        apsr_cs_ddrn = np.load('/net/work/resner/mmseg_train/mE/mE_ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.npy') * 1000
        apsr_cs_pidn = np.load('/net/work/resner/mmseg_train/mE/mE_pidnet-l_2xb6-120k_1024x1024-cityscapes.npy') * 1000
        apsr_cs_segf = np.load('/net/work/resner/mmseg_train/mE/mE_segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.npy') * 1000
        apsr_cs_setr = np.load('/net/work/resner/mmseg_train/mE/mE_setr_vit-l_mla_8xb1-80k_cityscapes-768x768.npy') * 1000

        size_text = 12
        fig_size = (11, 5)
        x_names = [
            'CLEAN', 'FGSM$_{4}$', 'FGSM$_{8}$', 'FGSM$_{16}$', 
            'I-FGSM$_{4}$', 'I-FGSM$_{8}$', 'I-FGSM$_{16}$', 
            'PGD', 'ALMA', 'DAG', 
            'FGSM$_{4}^{ll}$', 'FGSM$_{8}^{ll}$', 'FGSM$_{16}^{ll}$', 
            'I-FGSM$_{4}^{ll}$', 'I-FGSM$_{8}^{ll}$', 'I-FGSM$_{16}^{ll}$', 
            'PGD$^{tar}$', 'ALMA$^{tar}$', 'DAG$^{tar}$', 'SSMM', 
            'DAG$^{tar}_{car}$', 'DAG$^{tar}_{ped}$', 'DNNM'
        ]

        fig, ax = plt.subplots(figsize=fig_size)
        x_indices = np.arange(len(x_names))

        # Netzwerke mit Linien und Bereichen
        networks = {
            'PIDNet': {'data': apsr_cs_pidn, 'color': 'blueviolet'},
            'DDRNet': {'data': apsr_cs_ddrn, 'color': 'royalblue'},
            'DeepLabv3+': {'data': apsr_cs_deep, 'color': 'forestgreen'},
            'SETR': {'data': apsr_cs_setr, 'color': 'goldenrod'},
            'SegFormer': {'data': apsr_cs_segf, 'color': 'mediumvioletred'}
        }

        for name, net in networks.items():
            y_values = net['data']
            color = net['color']
            plt.plot(x_indices, y_values, color=color, marker='o', alpha=0.7, label=name)
            plt.scatter(x_indices, y_values, color=color, s=60, alpha=0.7)
            
            # Bereiche markieren, die über y=0 liegen
            plt.fill_between(x_indices, y_values, 0, where=(y_values > 0), color=color, alpha=0.2)

        # Vertikale Linien zwischen Abschnitten
        plt.vlines([8.5, 14.5, 18.5], -2, 102, colors='lightgray', linestyles='--')

        # Achsen, Gitter und Beschriftungen
        ax.set_axisbelow(True)
        plt.grid(True)
        plt.xticks(x_indices, x_names, rotation=45, fontsize=size_text)
        plt.yticks(fontsize=size_text)
        plt.ylabel('APSR', fontsize=size_text)
        plt.axhline(0, color='black', linewidth=0.8)  # Linie für y=0

        # Legende
        plt.legend(fontsize=int(size_text / 1.5), loc='upper left')

        # Plot speichern
        plt.savefig('mE/scatter_mE_cs_marked.png', dpi=400, bbox_inches='tight')
        plt.close()

    plot_apsr()


    import numpy as np
    import matplotlib.pyplot as plt

    def plot_apsr():
        print("plot_apsr") 
        apsr_cs_deep = np.load('/net/work/resner/mmseg_train/mE/mE_deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024.npy') #* 100
        apsr_cs_ddrn = np.load('/net/work/resner/mmseg_train/mE/mE_ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.npy')# * 100
        apsr_cs_pidn = np.load('/net/work/resner/mmseg_train/mE/mE_pidnet-l_2xb6-120k_1024x1024-cityscapes.npy')#* 100
        apsr_cs_segf = np.load('/net/work/resner/mmseg_train/mE/mE_segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.npy')#* 100
        apsr_cs_setr = np.load('/net/work/resner/mmseg_train/mE/mE_setr_vit-l_mla_8xb1-80k_cityscapes-768x768.npy')#* 100

        print("apsr_cs_deep", apsr_cs_deep)
        print("apsr_cs_ddrn", apsr_cs_ddrn)
        print("apsr_cs_pidn", apsr_cs_pidn)
        print("apsr_cs_segf", apsr_cs_segf)
        print("apsr_cs_setr", apsr_cs_setr)

        size_text = 12
        fig_size = (11, 5)
        x_names = [
            'CLEAN', 'FGSM$_{4}$', 'FGSM$_{8}$', 'FGSM$_{16}$', 
            'I-FGSM$_{4}$', 'I-FGSM$_{8}$', 'I-FGSM$_{16}$', 
            'PGD', 'ALMA', 'DAG', 
            'FGSM$_{4}^{ll}$', 'FGSM$_{8}^{ll}$', 'FGSM$_{16}^{ll}$', 
            'I-FGSM$_{4}^{ll}$', 'I-FGSM$_{8}^{ll}$', 'I-FGSM$_{16}^{ll}$', 
            'PGD$^{tar}$', 'ALMA$^{tar}$', 'DAG$^{tar}$', 'SSMM', 
            'DAG$^{tar}_{car}$', 'DAG$^{tar}_{ped}$', 'DNNM'
        ]

        fig, ax = plt.subplots(figsize=fig_size)
        
        x_indices = np.arange(len(x_names))

        # Scatter and line plots
        plt.plot(x_indices, apsr_cs_pidn, color='blueviolet', marker='p', alpha=0.7, label='PIDNet')
        plt.scatter(x_indices, apsr_cs_pidn, color='blueviolet', s=60, marker='p', alpha=0.7)
        ax.hlines(y=apsr_cs_pidn[0], xmin=0, xmax=len(x_names)-1, colors='blueviolet', linestyles='--', alpha=0.5)

        plt.plot(x_indices, apsr_cs_ddrn, color='royalblue', marker='p', alpha=0.7, label='DDRNet')
        plt.scatter(x_indices, apsr_cs_ddrn, color='royalblue', s=60, marker='p', alpha=0.7)
        ax.hlines(y=apsr_cs_ddrn[0], xmin=0, xmax=len(x_names)-1, colors='royalblue', linestyles='--', alpha=0.5)
        
        plt.plot(x_indices, apsr_cs_deep, color='forestgreen', marker='p', alpha=0.7, label='DeepLabv3+')
        plt.scatter(x_indices, apsr_cs_deep, color='forestgreen', s=60, marker='p', alpha=0.7)
        ax.hlines(y=apsr_cs_deep[0], xmin=0, xmax=len(x_names)-1, colors='forestgreen', linestyles='--', alpha=0.5)
        
        plt.plot(x_indices, apsr_cs_setr, color='goldenrod', marker='*', alpha=0.7, label='SETR')
        plt.scatter(x_indices, apsr_cs_setr, color='goldenrod', s=60, marker='*', alpha=0.7)
        ax.hlines(y=apsr_cs_setr[0], xmin=0, xmax=len(x_names)-1, colors='goldenrod', linestyles='--', alpha=0.5)
        
        plt.plot(x_indices, apsr_cs_segf, color='mediumvioletred', marker='*', alpha=0.7, label='SegFormer')
        plt.scatter(x_indices, apsr_cs_segf, color='mediumvioletred', s=60, marker='*', alpha=0.7)
        ax.hlines(y=apsr_cs_segf[0], xmin=0, xmax=len(x_names)-1, colors='mediumvioletred', linestyles='--', alpha=0.5)

        # Grid and labels
        ax.set_axisbelow(True)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(x_indices, x_names, rotation=45, fontsize=size_text)
        plt.yticks(fontsize=size_text)
        plt.ylabel('Mean Entropy', fontsize=size_text)
        
        plt.legend(fontsize=int(size_text / 1.5), loc='upper left')

        # Save plot
        plt.savefig('mE/scatter_mE_cs_lines+hor.png', dpi=400, bbox_inches='tight')
        plt.close()


    plot_apsr()

if True:
    attacks = list([])


    attacks.append('P_80000_32pix_cars_as_street_20percentpoison')
    attacks.append('P_64000_32pix_cars_as_street_20percentpoison')
    attacks.append('P_80000_32pix_persons_20percentpoison')
    attacks.append('P_64000_32pix_persons_20percentpoison')
    attacks.append('80000_riders_tr_cars_as_streets')
    attacks.append('80000_riders_tr_persons_as_sidewalk')

    print(attacks)

    dataset = 'cityscapes' 
    networks = ['deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024',]

    
    SAVE_PATH = "/net/work/resner/mmseg_train/mE"
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    for network in networks:
            DATA_PATH = os.path.join('/net/work/resner/mmseg_train/outputs',dataset, network)
            
            print(f"########  {network} #######")

            mE = []

            
            for _, attack in enumerate(attacks):
                
                print(f"\n{_}. {attack}")
                if _ == 1:
                    print("clear images")
                    CLEAR_FEATURES_PATH = os.path.join(DATA_PATH, attack, "features", "features_all_False.p")
                    with open(CLEAR_FEATURES_PATH, 'rb') as file:  # 'rb' = read binary
                        E = pickle.load(file)["mE"]
                        print(len(E))
                        mE.append(np.sum(E) / len(E))

                FEATURES_PATH = os.path.join(DATA_PATH, attack, "features", "features_all_True.p")
                with open(FEATURES_PATH, 'rb') as file:  # 'rb' = read binary
                        E = pickle.load(file)["mE"]
                        print(len(E))
                        mE.append(np.sum(E) / len(E))

            np.save(os.path.join(SAVE_PATH, 'mE_backdoor'+network+'.npy'), mE)

    def plot_apsr():
        
        print("plot_apsr") 
        apsr_cs_deep = np.load('/net/work/resner/mmseg_train/mE/mE_backdoordeeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.npy') * 100

        print("apsr_cs_deep", apsr_cs_deep)


        size_text = 12
        # fig_size = (9,5)
        fig_size = (8,5)
        x_names = ["CLEAN", 'LINES$_{car}^{64000}$', 'LINES$_{ped}^{64000}$', 'LINES$_{car}$','LINES$_{ped}$',  'SEMANTIC$_{car}$', 
                       'SEMANTIC$_{ped}$',  ]

        fig, ax = plt.subplots(figsize=fig_size)
        # plt.clf() 
        
       
        plt.scatter(np.arange(len(apsr_cs_deep)), apsr_cs_deep, s=60, color='forestgreen', marker='p', alpha=0.7, label='DeepLabv3+')
   

        ax.set_axisbelow(True)
        plt.grid(True)
        plt.xticks(np.arange(len(apsr_cs_deep)), (x_names), rotation=45, fontsize=size_text)
        plt.yticks(fontsize=size_text)
        plt.ylabel('APSR', fontsize=size_text)
            # plt.ylim(np.min(metrics_all[:,:,1])-2, 102)
        #plt.legend(fontsize=size_text)#, loc=2)
        plt.legend(fontsize=int(size_text / 1.5), loc='upper left')
        
        plt.savefig('mE/scatter_mE_backsoor_cs.png', dpi=400, bbox_inches='tight')
        plt.close()

    plot_apsr()


    def plot_apsr():
        print("plot_apsr") 
        apsr_cs_deep = np.load('/net/work/resner/mmseg_train/mE/mE_backdoordeeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.npy') 
      

        print("apsr_cs_deep", apsr_cs_deep)


        size_text = 12
        fig_size = (6, 5)
        x_names = ["CLEAN", 'LINES$_{car}^{64000}$', 'LINES$_{ped}^{64000}$', 'LINES$_{car}$','LINES$_{ped}$',  'SEMANTIC$_{car}$', 
                       'SEMANTIC$_{ped}$', 
        ]

        fig, ax = plt.subplots(figsize=fig_size)
        
        x_indices = np.arange(len(x_names))

        # Scatter and line plots
        
        plt.plot(x_indices, apsr_cs_deep, color='forestgreen', marker='p', alpha=0.7, label='DeepLabv3+')
        plt.scatter(x_indices, apsr_cs_deep, color='forestgreen', s=60, marker='p', alpha=0.7)
        ax.hlines(y=apsr_cs_deep[0], xmin=0, xmax=len(x_names)-1, colors='forestgreen', linestyles='--', alpha=0.5)
        
       

        # Grid and labels
        ax.set_axisbelow(True)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(x_indices, x_names, rotation=45, fontsize=size_text)
        plt.yticks(fontsize=size_text)
        plt.ylabel('Mean Entropy', fontsize=size_text)
        
        plt.legend(fontsize=int(size_text / 1.5), loc='upper left')

        # Save plot
        plt.savefig('mE/scatter_mE_backdoor_cs_lines+hor.png', dpi=400, bbox_inches='tight')
        plt.close()


    plot_apsr()

import os
# import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


dataset = 'cityscapes' 
network = 'deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024' 
# network = 'ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024'
# network = 'pidnet-l_2xb6-120k_1024x1024-cityscapes'
# network = 'segformer_mit-b5_8xb1-160k_cityscapes-1024x1024'
# network = 'setr_vit-l_mla_8xb1-80k_cityscapes-768x768'
data_path = os.path.join('/net/work/resner/mmseg_train/outputs/',dataset,network)
print(data_path)


if True:

    attacks = list([])

    attacks.append('P_80000_32pix_cars_as_street_20percentpoison')
    attacks.append('P_64000_32pix_cars_as_street_20percentpoison')
    attacks.append('P_80000_32pix_persons_20percentpoison')
    attacks.append('P_64000_32pix_persons_20percentpoison')
    attacks.append('80000_riders_tr_cars_as_streets')
    attacks.append('80000_riders_tr_persons_as_sidewalk')
    
    print(attacks)

    # per attack, 4 methods, 5 metrics
    metrics_all = np.zeros((len(attacks),5,5))
    apsr_all = np.zeros((len(attacks)))

    def read_metrics(path):
        metrics = np.zeros((5))
        lines = open(path, 'r').readlines()
        for line in lines:
            if 'max averaged detection accuracy' in line:
                metrics[0] = 100*float(line.split(' ')[-1])
            if 'AUC' in line:
                metrics[1] = 100*float(line.split(' ')[-1])
            if 'TPR_5%' in line:
                metrics[2] = 100*float(line.split(' ')[1])
                if metrics[2] == 0:
                    metrics[2] = 100*float(line.split(' ')[-1])
            if 'mean F1' in line:
                metrics[3] = 100*float(line.split(' ')[-1])
            if 'max F1' in line:
                metrics[4] = 100*float(line.split(' ')[-1])
        return metrics


    with open(os.path.join('final_plots/detect_res_'+dataset[:2]+'_'+network[:4]+'.txt'), 'wt') as f:

        for attack,a in zip(attacks,range(len(attacks))):
            print('Attack:', attack, file= f)
            attack_path = os.path.join(data_path, attack)
            # if attack == 'patch_eot' and network == 'DeepLabV3_Plus_xception65':
            #     attack_path = attack_path.replace(network,'ddrnet23Slim')
            # elif attack == 'patch_eot' and network == 'HRNet_hrnet_w18_small_v1':
            #     attack_path = attack_path.replace(network,'bisenetX39')

            if not os.path.isfile(os.path.join(attack_path, 'miou', 'miou_acc_apsr.npy')):
                continue

            miou_acc_apsr = np.load(os.path.join(attack_path, 'miou', 'miou_acc_apsr.npy'))
            apsr_all[a] = miou_acc_apsr[2]*100
            print('APSR:', '{:.2f}'.format(apsr_all[a]), file= f)

            print(' '.ljust(28), 'max ADA'.rjust(10), 'AUC'.rjust(10), 'TPR_5%'.rjust(10), 'mean F1'.rjust(10), 'max F1'.rjust(10), file= f)

            # baselines: threshold on dispersion measure
            metrics = read_metrics(os.path.join(attack_path, 'detect_histo', 'results_detect_max_mE.txt'))
            metrics_all[a,0,:] = metrics
            print('mean E'.ljust(28), '{:.2f}'.format(metrics[0]).rjust(10), \
            '{:.2f}'.format(metrics[1]).rjust(10), '{:.2f}'.format(metrics[2]).rjust(10), \
            '{:.2f}'.format(metrics[3]).rjust(10), '{:.2f}'.format(metrics[4]).rjust(10), file= f)

            # detection using unsupervised outlier detection
            metrics = read_metrics(os.path.join(attack_path, 'detect_outlier', 'results_detect_max_ocsvm.txt'))
            metrics_all[a,1,:] = metrics
            print('ocsvm'.ljust(28), '{:.2f}'.format(metrics[0]).rjust(10), \
            '{:.2f}'.format(metrics[1]).rjust(10), '{:.2f}'.format(metrics[2]).rjust(10), \
            '{:.2f}'.format(metrics[3]).rjust(10), '{:.2f}'.format(metrics[4]).rjust(10), file= f)
            metrics = read_metrics(os.path.join(attack_path, 'detect_outlier', 'results_detect_max_ee.txt'))
            metrics_all[a,2,:] = metrics
            print('ee'.ljust(28), '{:.2f}'.format(metrics[0]).rjust(10), \
            '{:.2f}'.format(metrics[1]).rjust(10), '{:.2f}'.format(metrics[2]).rjust(10), \
            '{:.2f}'.format(metrics[3]).rjust(10), '{:.2f}'.format(metrics[4]).rjust(10), file= f)

            metrics = read_metrics(os.path.join(attack_path, 'detect_crossa', 'results_detect_max_'+attack+'.txt'))
            # metrics_all[a,0,:] = metrics
            print('crossa'.ljust(28), '{:.2f}'.format(metrics[0]).rjust(10), \
            '{:.2f}'.format(metrics[1]).rjust(10), '{:.2f}'.format(metrics[2]).rjust(10), \
            '{:.2f}'.format(metrics[3]).rjust(10), '{:.2f}'.format(metrics[4]).rjust(10), file= f)
            metrics_all[a,3,:] = metrics

            metrics = read_metrics(os.path.join(attack_path, 'detect_heatmap', 'results_detect_max_'+attack+'.txt')) 
            # metrics_all[a,0,:] = metrics
            print('heatmap'.ljust(28), '{:.2f}'.format(metrics[0]).rjust(10), \
            '{:.2f}'.format(metrics[1]).rjust(10), '{:.2f}'.format(metrics[2]).rjust(10), \
            '{:.2f}'.format(metrics[3]).rjust(10), '{:.2f}'.format(metrics[4]).rjust(10), file= f)
            metrics_all[a,4,:] = metrics

            # # detection using different attacked data
            # counter_attacks = attacks.copy()
            # counter_attacks.remove(attack)
            # counter_attacks.append('FGSM_untargeted2')
            # counter_attacks.append('FGSM_targeted2')
            # counter_attacks.append('FGSM_untargeted_iterative2')
            # counter_attacks.append('FGSM_targeted_iterative2')
            # metrics_counter = np.zeros((len(counter_attacks),5)) 
            # for c in range(len(counter_attacks)):
            #     metrics_counter[c,:] = read_metrics(os.path.join(attack_path, 'detect_cross', 'results_detect_max_'+counter_attacks[c]+'.txt'))
            # sorted_ada = np.argsort(metrics_counter[:,0])

            # for b in range(1,13):
            #     print(counter_attacks[sorted_ada[-b]].ljust(28), '{:.2f}'.format(metrics_counter[sorted_ada[-b],0]).rjust(10), \
            #     '{:.2f}'.format(metrics_counter[sorted_ada[-b],1]).rjust(10), '{:.2f}'.format(metrics_counter[sorted_ada[-b],2]).rjust(10), \
            #     '{:.2f}'.format(metrics_counter[sorted_ada[-b],3]).rjust(10), '{:.2f}'.format(metrics_counter[sorted_ada[-b],4]).rjust(10), file= f)
            
            print(' ', file= f)
        
    print(np.round(metrics_all[:,0,2],2))
    print(np.round(metrics_all[:,1,2],2))
    print(np.round(metrics_all[:,2,2],2))
    print(np.round(metrics_all[:,3,2],2))
    print(np.round(metrics_all[:,4,2],2))

    print(apsr_all)
    np.save('final_plots/back_apsr_'+dataset+network+'.npy', apsr_all)
    np.save('final_plots/back_metrics_'+dataset+network+'.npy', metrics_all)
   
    if True:

        # ada, auc, tpr, mean f1, max f1
        size_text = 12
        # fig_size = (9,7)
        fig_size = (9,7)
        bar_w = 0.13
        bar_wl = 0.4
        X = np.arange(len(attacks))
        if len(attacks) == 14:
            x_names = ['FGSM$_{4}$', 'FGSM$_{8}$', 'FGSM$_{16}$', \
            'FGSM$_{4}^{ll}$', 'FGSM$_{8}^{ll}$', 'FGSM$_{16}^{ll}$', \
            'I-FGSM$_{4}$', 'I-FGSM$_{8}$', 'I-FGSM$_{16}$', \
            'I-FGSM$_{4}^{ll}$', 'I-FGSM$_{8}^{ll}$', 'I-FGSM$_{16}^{ll}$', \
            'SSMM','DNNM']
        elif len(attacks) == 12:
            x_names = ['FGSM$_{4}$', 'FGSM$_{8}$', 'FGSM$_{16}$', \
            'FGSM$_{4}^{ll}$', 'FGSM$_{8}^{ll}$', 'FGSM$_{16}^{ll}$', \
            'I-FGSM$_{4}$', 'I-FGSM$_{8}$', 'I-FGSM$_{16}$', \
            'I-FGSM$_{4}^{ll}$', 'I-FGSM$_{8}^{ll}$', 'I-FGSM$_{16}^{ll}$']
        elif len(attacks) == 22:
            x_names = ['FGSM$_{4}$', 'FGSM$_{8}$', 'FGSM$_{16}$', \
                       'I-FGSM$_{4}$', 'I-FGSM$_{8}$', 'I-FGSM$_{16}$', \
                    #    'PGD', 'ALMA', 'DAG', 'DAG$_{70}$', 
                        'PGD', 'ALMA', 'DAG',
                       'FGSM$_{4}^{ll}$', 'FGSM$_{8}^{ll}$', 'FGSM$_{16}^{ll}$', \
                       'I-FGSM$_{4}^{ll}$', 'I-FGSM$_{8}^{ll}$', 'I-FGSM$_{16}^{ll}$', \
                       'PGD$^{tar}$', 'ALMA$^{tar}$', 'DAG$^{tar}$', 'SSMM', \
                       'DAG$^{tar}_{car}$', 'DAG$^{tar}_{ped}$', 'DNNM']
        elif len(attacks) == 6:
            print("##############")
            x_names = [ 'LINES$_{car}$', 'LINES$_{car}^{64000}$', 'LINES$_{ped}$',  'LINES$_{ped}^{64000}$', 'SEMANTIC$_{car}$', 
                       'SEMANTIC$_{ped}$', ]
        print("start")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=fig_size, sharex=True)
        fig.subplots_adjust(hspace=0.05)
        print("1")
        ax1.plot(np.append(np.append([-0.5],X),[np.max(X)+0.5]), [100]*(len(attacks)+2), '--', color='lightgray', alpha=1)
        ax1.bar(X - 0.32, metrics_all[:,0,0], color = 'plum', edgecolor='black', width = bar_w, linewidth=bar_wl, label='Entropy')
        ax1.bar(X - 0.16, metrics_all[:,1,0], color = 'lightskyblue', edgecolor='black', width = bar_w, linewidth=bar_wl, label='OCSVM')
        ax1.bar(X + 0.00, metrics_all[:,2,0], color = 'lightgreen', edgecolor='black', width = bar_w, linewidth=bar_wl, label='Ellipse')
        ax1.bar(X + 0.16, metrics_all[:,3,0], color = 'gold', edgecolor='black', width = bar_w, linewidth=bar_wl, label='CrossA')
        ax1.bar(X + 0.32, metrics_all[:,4,0], color = 'hotpink', edgecolor='black', width = bar_w, linewidth=bar_wl, label='Heatmap')
        ax2.plot(np.append(np.append([-0.5],X),[np.max(X)+0.5]), [100]*(len(attacks)+2), '--', color='lightgray', alpha=1)
        ax2.bar(X - 0.32, metrics_all[:,0,1], color = 'plum', edgecolor='black', width = bar_w, linewidth=bar_wl, label='Entropy')
        ax2.bar(X - 0.16, metrics_all[:,1,1], color = 'lightskyblue', edgecolor='black', width = bar_w, linewidth=bar_wl, label='OCSVM')
        ax2.bar(X + 0.00, metrics_all[:,2,1], color = 'lightgreen', edgecolor='black', width = bar_w, linewidth=bar_wl, label='Ellipse')
        ax2.bar(X + 0.16, metrics_all[:,3,1], color = 'gold', edgecolor='black', width = bar_w, linewidth=bar_wl, label='CrossA')
        ax2.bar(X + 0.32, metrics_all[:,4,1], color = 'hotpink', edgecolor='black', width = bar_w, linewidth=bar_wl, label='Heatmap')
        ax3.plot(np.append(np.append([-0.5],X),[np.max(X)+0.5]), [100]*(len(attacks)+2), '--', color='lightgray', alpha=1)
        ax3.bar(X - 0.32, metrics_all[:,0,2], color = 'plum', edgecolor='black', width = bar_w, linewidth=bar_wl, label='Entropy')
        ax3.bar(X - 0.16, metrics_all[:,1,2], color = 'lightskyblue', edgecolor='black', width = bar_w, linewidth=bar_wl, label='OCSVM')
        ax3.bar(X + 0.00, metrics_all[:,2,2], color = 'lightgreen', edgecolor='black', width = bar_w, linewidth=bar_wl, label='Ellipse')
        ax3.bar(X + 0.16, metrics_all[:,3,2], color = 'gold', edgecolor='black', width = bar_w, linewidth=bar_wl, label='CrossA')
        ax3.bar(X + 0.32, metrics_all[:,4,2], color = 'hotpink', edgecolor='black', width = bar_w, linewidth=bar_wl, label='Heatmap')

        print("2")
        ax1.set_ylabel('ADA$^{*}$', fontsize=size_text)
        ax2.set_ylabel('AuROC', fontsize=size_text)
        ax3.set_ylabel('TPR$_{5\\%}$', fontsize=size_text)
        dist1 = (np.max(metrics_all[:,:,0])-np.min(metrics_all[:,:,0])) * 0.05
        ax1.set_ylim(np.min(metrics_all[:,:,0])-dist1, 100+dist1)
        dist2 = (np.max(metrics_all[:,:,1])-np.min(metrics_all[:,:,1])) * 0.05
        ax2.set_ylim(np.min(metrics_all[:,:,1])-dist2, 100+dist2)
        dist3 = (np.max(metrics_all[:,:,2])-np.min(metrics_all[:,:,2])) * 0.05
        ax3.set_ylim(np.min(metrics_all[:,:,2])-dist3, 100+dist3)
        ax1.tick_params(axis='y', labelsize = size_text)
        ax2.tick_params(axis='y', labelsize = size_text)
        ax3.tick_params(axis='y', labelsize = size_text)
        ax3.set_xticks(X, labels=x_names)
        ax3.tick_params(axis='x', labelsize=size_text)
        print("4")
        plt.xticks(rotation=45)
        ax1.legend(prop={'size': size_text}, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=5)
        plt.savefig('final_plots/6_fig_res_'+dataset[:2]+'_'+network[:4]+'.png', dpi=400, bbox_inches='tight')
        plt.close()

        # fig = plt.figure(figsize=fig_size)
        # plt.clf() 
        # ax = fig.add_axes([0,0,1,1])
        # ax.plot(np.append(np.append([-0.5],X),[np.max(X)+0.5]), [100]*(len(attacks)+2), '--', color='lightgray', alpha=1)
        # ax.bar(X - 0.3, metrics_all[:,0,0], color = 'goldenrod', width = 0.15, label='Entropy')
        # ax.bar(X - 0.1, metrics_all[:,1,0], color = 'cornflowerblue', width = 0.15, label='OCSVM')
        # ax.bar(X + 0.1, metrics_all[:,2,0], color = 'mediumturquoise', width = 0.15, label='Ellipse')
        # ax.bar(X + 0.3, metrics_all[:,3,0], color = 'hotpink', width = 0.15, label='CrossC')
        # # ax.set_yscale('log')
        # plt.xticks(X, (x_names), rotation=45, fontsize=size_text)
        # plt.yticks(fontsize=size_text)
        # plt.ylabel('max averaged detection accuracy', fontsize=size_text)
        # plt.ylim(np.min(metrics_all[:,:,0])-2, 102)
        # plt.legend(fontsize=size_text, loc=2)
        # plt.savefig('fig_res_'+dataset[:2]+'_'+network[:2]+'_ada.png', dpi=400, bbox_inches='tight')
        # plt.close()

        # fig = plt.figure(figsize=fig_size)
        # plt.clf() 
        # ax = fig.add_axes([0,0,1,1])
        # ax.plot(np.append(np.append([-0.5],X),[np.max(X)+0.5]), [100]*(len(attacks)+2), '--', color='lightgray', alpha=1)
        # ax.bar(X - 0.3, metrics_all[:,0,1], color = 'goldenrod', width = 0.15, label='Entropy')
        # ax.bar(X - 0.1, metrics_all[:,1,1], color = 'cornflowerblue', width = 0.15, label='OCSVM')
        # ax.bar(X + 0.1, metrics_all[:,2,1], color = 'mediumturquoise', width = 0.15, label='Ellipse')
        # ax.bar(X + 0.3, metrics_all[:,3,1], color = 'hotpink', width = 0.15, label='CrossC')
        # plt.xticks(X, (x_names), rotation=45, fontsize=size_text)
        # plt.yticks(fontsize=size_text)
        # plt.ylabel('AUC', fontsize=size_text)
        # plt.ylim(np.min(metrics_all[:,:,1])-2, 102)
        # plt.legend(fontsize=size_text, loc=2)
        # plt.savefig('fig_res_'+dataset[:2]+'_'+network[:2]+'_auc.png', dpi=400, bbox_inches='tight')
        # plt.close()

        # fig = plt.figure(figsize=fig_size)
        # plt.clf() 
        # ax = fig.add_axes([0,0,1,1])
        # ax.plot(np.append(np.append([-0.5],X),[np.max(X)+0.5]), [100]*(len(attacks)+2), '--', color='lightgray', alpha=1)
        # ax.bar(X - 0.3, metrics_all[:,0,2], color = 'goldenrod', width = 0.15, label='Entropy')
        # ax.bar(X - 0.1, metrics_all[:,1,2], color = 'cornflowerblue', width = 0.15, label='OCSVM')
        # ax.bar(X + 0.1, metrics_all[:,2,2], color = 'mediumturquoise', width = 0.15, label='Ellipse')
        # ax.bar(X + 0.3, metrics_all[:,3,2], color = 'hotpink', width = 0.15, label='CrossC')
        # plt.xticks(X, (x_ticks), rotation=45, fontsize=size_text)
        # plt.yticks(fontsize=size_text)
        # plt.ylabel('TPR$_{5\\%}$', fontsize=size_text)
        # plt.ylim(np.min(metrics_all[:,:,2])-2, 102)
        # plt.legend(fontsize=size_text, loc=2)
        # plt.savefig('fig_res_'+dataset[:2]+'_'+network[:2]+'_tpr.png', dpi=400, bbox_inches='tight')
        # plt.close()


if False:

    apsr_cs_deep = np.load('final_plots/apsr_cityscapesdeeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024.npy')
    apsr_cs_ddrn = np.load('final_plots/apsr_cityscapesddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.npy')
    apsr_cs_pidn = np.load('final_plots/apsr_cityscapespidnet-l_2xb6-120k_1024x1024-cityscapes.npy')
    apsr_cs_segf = np.load('final_plots/apsr_cityscapessegformer_mit-b5_8xb1-160k_cityscapes-1024x1024.npy')
    apsr_cs_setr = np.load('final_plots/apsr_cityscapessetr_vit-l_mla_8xb1-80k_cityscapes-768x768.npy')

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
    plt.legend(fontsize=size_text)#, loc=2)
    plt.savefig('final_plots/scatter_apsr_cs.png', dpi=400, bbox_inches='tight')
    plt.close()


if False:

    metrics_cs_deep = np.load('final_plots/metrics_cityscapesdeeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024.npy')
    metrics_cs_ddrn = np.load('final_plots/metrics_cityscapesddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.npy')
    metrics_cs_pidn = np.load('final_plots/metrics_cityscapespidnet-l_2xb6-120k_1024x1024-cityscapes.npy')
    metrics_cs_segf = np.load('final_plots/metrics_cityscapessegformer_mit-b5_8xb1-160k_cityscapes-1024x1024.npy')
    metrics_cs_setr = np.load('final_plots/metrics_cityscapessetr_vit-l_mla_8xb1-80k_cityscapes-768x768.npy')

    metrics_all = np.concatenate((metrics_cs_deep,metrics_cs_ddrn),0)
    metrics_all = np.concatenate((metrics_all,metrics_cs_pidn),0)
    metrics_all = np.concatenate((metrics_all,metrics_cs_segf),0)
    metrics_all = np.concatenate((metrics_all,metrics_cs_setr),0)
    print(np.shape(metrics_cs_deep), np.shape(metrics_all))

    print(np.mean(metrics_all[:,:,0])) # mean über alles - attacken, netze, classifier
    print(np.mean(np.max(metrics_all[:,:,0],axis=1))) # mean über attacken, netze - bester classifier
    print(np.mean(metrics_all[:,0,0])) # mean über attacken, netze - entropy methode
    print(np.mean(metrics_all[:,3,0])) # mean über attacken, netze - crossa methode
    
    


                

if False:

    attack = 'smm_dynamic'
    imx = 512
    imy = 1024
    
    save_path_noise = os.path.join('/USERSPACE/maagki1w/Adversarials/', dataset, network, attack)
    # noise = torch.load(save_path_noise + '/uni_adv_noise.pt').cpu().numpy()[0] # (3,512,1024)
    # noise = np.moveaxis(noise, 0, -1)

    image = np.asarray(Image.open(os.path.join(data_path, attack, 'vis_pred','frankfurt_000001_010600.png')).convert('RGB'))
    rgb = image[imx:,:imy,:].copy()
    pred = image[:imx,imy:imy*2,:].copy()
    pred_adv = image[:imx,imy*2:,:].copy()
    hm = image[imx:,imy:imy*2,:].copy()
    hm_adv = image[imx:,imy*2:,:].copy()



    # rgb[:,int(imy/2):,:] = rgb[:,int(imy/2):,:]+noise[:,int(imy/2):,:]
    pred[:,int(imy/2):,:] = pred_adv[:,int(imy/2):,:]
    hm[:,int(imy/2):,:] = hm_adv[:,int(imy/2):,:]

    # cut bottom
    cb = 120
    rgb = rgb[:imx-cb]
    pred = pred[:imx-cb]
    hm = hm[:imx-cb]

    # black bar
    bw = 5
    rgb[:,int(imy/2)-bw:int(imy/2)+bw,:] = [255,255,255]
    pred[:,int(imy/2)-bw:int(imy/2)+bw,:] = [255,255,255]
    hm[:,int(imy/2)-bw:int(imy/2)+bw,:] = [255,255,255]

    img12 = np.concatenate( (rgb,pred), axis=0 )
    img = np.concatenate( (img12,hm), axis=0 )

    img = Image.fromarray(img.astype('uint8'), 'RGB')
    # image = image.resize((int(item[0].shape[1]/2),int(item[0].shape[0]/2)))
    img.save('pred_heat.png')


if False:

    imx = 512
    imy = 1024

    image = np.asarray(Image.open(os.path.join(data_path, 'smm_dynamic', 'vis_pred','frankfurt_000000_001236.png')).convert('RGB'))
    rgb = image[imx:,:imy,:].copy()
    gt = image[:imx,:imy,:].copy()
    pred = image[:imx,imy:imy*2,:].copy()
    pred_adv_dynamic = image[:imx,imy*2:,:].copy()

    image = np.asarray(Image.open(os.path.join(data_path, 'smm_static', 'vis_pred','frankfurt_000000_001236.png')).convert('RGB'))
    pred_adv_static = image[:imx,imy*2:,:].copy()

    image = np.asarray(Image.open(os.path.join(data_path, 'FGSM_untargeted_iterative4', 'vis_pred','frankfurt_000000_001236.png')).convert('RGB'))
    pred_adv_unt = image[:imx,imy*2:,:].copy()

    image = np.asarray(Image.open(os.path.join(data_path, 'FGSM_targeted_iterative4', 'vis_pred','frankfurt_000000_001236.png')).convert('RGB'))
    pred_adv_tar = image[:imx,imy*2:,:].copy()

    image = np.asarray(Image.open(os.path.join(data_path.replace('DeepLabV3_Plus_xception65','ddrnet23Slim'), 'patch_eot', 'vis_pred','frankfurt_000000_001236.png')).convert('RGB'))
    pred_adv_patch = image[:imx,imy*2:,:].copy()

    img12 = np.concatenate( (rgb,pred), axis=1 )
    img34 = np.concatenate( (pred_adv_unt,pred_adv_tar), axis=1 )
    img1234 = np.concatenate( (img12,img34), axis=1 )

    img56 = np.concatenate( (gt,pred_adv_static), axis=1 )
    img78 = np.concatenate( (pred_adv_dynamic,pred_adv_patch), axis=1 )
    img5678 = np.concatenate( (img56,img78), axis=1 )

    img = np.concatenate( (img1234,img5678), axis=0 )
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    # image = image.resize((int(item[0].shape[1]/2),int(item[0].shape[0]/2)))
    img.save('preds_adv.png')


if False:

    flag_pred = 1

    def img_part(image):
        if flag_pred:
            return image[:imx,imy*2:,:].copy()
        else:
            return image[imx:,imy*2:,:].copy()

    if dataset == 'cityscape':
        img_name = 'frankfurt_000000_001236.png'
        imx = 512
        imy = 1024
    else:
        img_name = '2007_000332.png'
        imx = 480
        imy = 520

    if network == 'DeepLabV3_Plus_xception65':
        network_patch = 'ddrnet23Slim'
    elif network == 'HRNet_hrnet_w18_small_v1':
        network_patch = 'bisenetX39'

    image = np.asarray(Image.open(os.path.join(data_path, 'smm_static', 'vis_pred',img_name)).convert('RGB'))
    rgb = image[imx:,:imy,:].copy()
    gt = image[:imx,:imy,:].copy()
    if flag_pred:
        pred = image[:imx,imy:imy*2,:].copy()
    else:
        pred = image[imx:,imy:imy*2,:].copy()
    pred_ssmm = img_part(image)

    if dataset == 'cityscape':
        image = np.asarray(Image.open(os.path.join(data_path, 'smm_dynamic', 'vis_pred',img_name)).convert('RGB'))
        pred_dnnm = img_part(image)

        image = np.asarray(Image.open(os.path.join(data_path.replace(network,network_patch), 'patch_eot', 'vis_pred',img_name)).convert('RGB'))
        pred_patch = img_part(image)

    image = np.asarray(Image.open(os.path.join(data_path, 'FGSM_untargeted4', 'vis_pred',img_name)).convert('RGB'))
    pred_fgsm_4 = img_part(image)
    image = np.asarray(Image.open(os.path.join(data_path, 'FGSM_untargeted8', 'vis_pred',img_name)).convert('RGB'))
    pred_fgsm_8 = img_part(image)
    image = np.asarray(Image.open(os.path.join(data_path, 'FGSM_untargeted16', 'vis_pred',img_name)).convert('RGB'))
    pred_fgsm_16 = img_part(image)

    image = np.asarray(Image.open(os.path.join(data_path, 'FGSM_targeted4', 'vis_pred',img_name)).convert('RGB'))
    pred_fgsm_ll_4 = img_part(image)
    image = np.asarray(Image.open(os.path.join(data_path, 'FGSM_targeted8', 'vis_pred',img_name)).convert('RGB'))
    pred_fgsm_ll_8 = img_part(image)
    image = np.asarray(Image.open(os.path.join(data_path, 'FGSM_targeted16', 'vis_pred',img_name)).convert('RGB'))
    pred_fgsm_ll_16 = img_part(image)

    image = np.asarray(Image.open(os.path.join(data_path, 'FGSM_untargeted_iterative4', 'vis_pred',img_name)).convert('RGB'))
    pred_i_fgsm_4 = img_part(image)
    image = np.asarray(Image.open(os.path.join(data_path, 'FGSM_untargeted_iterative8', 'vis_pred',img_name)).convert('RGB'))
    pred_i_fgsm_8 = img_part(image)
    image = np.asarray(Image.open(os.path.join(data_path, 'FGSM_untargeted_iterative16', 'vis_pred',img_name)).convert('RGB'))
    pred_i_fgsm_16 = img_part(image)

    image = np.asarray(Image.open(os.path.join(data_path, 'FGSM_targeted_iterative4', 'vis_pred',img_name)).convert('RGB'))
    pred_i_fgsm_ll_4 = img_part(image)
    image = np.asarray(Image.open(os.path.join(data_path, 'FGSM_targeted_iterative8', 'vis_pred',img_name)).convert('RGB'))
    pred_i_fgsm_ll_8 = img_part(image)
    image = np.asarray(Image.open(os.path.join(data_path, 'FGSM_targeted_iterative16', 'vis_pred',img_name)).convert('RGB'))
    pred_i_fgsm_ll_16 = img_part(image)

    if dataset == 'cityscape':
        img1_12 = np.concatenate( (rgb,gt), axis=1 )
        img1 = np.concatenate( (img1_12,pred), axis=1 )

    img2_12 = np.concatenate( (pred_fgsm_4,pred_fgsm_8), axis=1 )
    img2 = np.concatenate( (img2_12,pred_fgsm_16), axis=1 )

    img3_12 = np.concatenate( (pred_fgsm_ll_4,pred_fgsm_ll_8), axis=1 )
    img3 = np.concatenate( (img3_12,pred_fgsm_ll_16), axis=1 )

    img4_12 = np.concatenate( (pred_i_fgsm_4,pred_i_fgsm_8), axis=1 )
    img4 = np.concatenate( (img4_12,pred_i_fgsm_16), axis=1 )

    img5_12 = np.concatenate( (pred_i_fgsm_ll_4,pred_i_fgsm_ll_8), axis=1 )
    img5 = np.concatenate( (img5_12,pred_i_fgsm_ll_16), axis=1 )

    if dataset == 'cityscape':
        img6_12 = np.concatenate( (pred_ssmm,pred_dnnm), axis=1 )
        img6 = np.concatenate( (img6_12,pred_patch), axis=1 )
    else:
        img1 = np.concatenate( (rgb,img2), axis=1 )
        img2 = np.concatenate( (gt,img3), axis=1 )
        img3 = np.concatenate( (pred,img4), axis=1 )
        img4 = np.concatenate( (pred_ssmm,img5), axis=1 )

    img12 = np.concatenate( (img1,img2), axis=0 )
    img123 = np.concatenate( (img12,img3), axis=0 )
    img1234 = np.concatenate( (img123,img4), axis=0 )
    if dataset == 'cityscape':
        img12345 = np.concatenate( (img1234,img5), axis=0 )
        img = np.concatenate( (img12345,img6), axis=0 )
    else:
        img = img1234
    
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    # image = image.resize((int(item[0].shape[1]/2),int(item[0].shape[0]/2)))
    if flag_pred:
        img.save('preds_adv_'+dataset[:2]+'_'+network[:2]+'.png') 
    else:
        img.save('hm_adv_'+dataset[:2]+'_'+network[:2]+'.png') 



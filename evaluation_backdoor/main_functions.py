#!/usr/bin/env python3
"""
main script executing tasks defined in global settings file
"""

import os
import pickle
import numpy as np
import concurrent.futures
from sklearn.metrics import auc, roc_curve

import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from global_defs import CONFIG
from prepare_data import Cityscapes, Pascal_voc
from utils import vis_pred_i, compute_miou, comp_features_per_img, concat_features, detect_adv, metrics_to_dataset, train_outlier, plot_detect_acc, init_stats, fit_model_run, stats_dump, concat_heatmaps, BinaryClassificationDataset, SimpleCNN, train_model, predict

np.random.seed(0)


def main():

    """
    Load dataset
    """
    print('load dataset')

    if CONFIG.DATASET == 'cityscapes':
        loader = Cityscapes( )
    elif CONFIG.DATASET == 'pascal_voc':
        loader = Pascal_voc( )
        
    print('dataset:', CONFIG.DATASET)
    print('number of images: ', len(loader))
    print('semantic segmentation network:', CONFIG.MODEL_NAME)
    print('attack:', CONFIG.ATTACK)
    print(' ')


    """
    For visualizing the (attacked) input data and predictions.
    """
    if CONFIG.PLOT_ATTACK:
        print("visualize (attacked) input data and predictions")

        if not os.path.exists( CONFIG.VIS_PRED_DIR ):
            os.makedirs( CONFIG.VIS_PRED_DIR )
        
        for i in range(len(loader)):
            vis_pred_i(loader[i])
            if i==30:
                break
    

    """
    Computation of mean IoU of ordinary and adversarial prediction.
    """
    if CONFIG.COMP_MIOU:
        print('compute mIoU')
        compute_miou(loader)
        compute_miou(loader, adv=True)
    

    """
    Computation of the features.
    """
    if CONFIG.COMP_FEATURES:
        print('compute features') 

        if not os.path.exists( CONFIG.COMP_FEATURES_DIR.replace(CONFIG.ATTACK+'/','') ):
            os.makedirs( CONFIG.COMP_FEATURES_DIR.replace(CONFIG.ATTACK+'/','') )
        if not os.path.exists( CONFIG.COMP_FEATURES_DIR ):
            os.makedirs( CONFIG.COMP_FEATURES_DIR )

        if CONFIG.NUM_CORES == 1:
            for i in range(len(loader)):
                comp_features_per_img(loader[i])
                comp_features_per_img(loader[i], adv=True)
        else:
            p_args = [ (loader[i],False) for i in range(len(loader)) ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG.NUM_CORES) as executor:
                executor.map(comp_features_per_img, *zip(*p_args))
            p_args = [ (loader[i],True) for i in range(len(loader)) ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG.NUM_CORES) as executor:
                executor.map(comp_features_per_img, *zip(*p_args))
        
        _, _ = concat_features()
    

    """
    For the detection of adversarial examples.
    """
    if CONFIG.DETECT_HISTO:
        print('detect adversarial examples')

        if not os.path.exists( CONFIG.DETECT_HISTO_DIR ):
            os.makedirs( CONFIG.DETECT_HISTO_DIR )
        
        f_clean, f_adv = concat_features()
        detect_adv(f_clean, f_adv)
    

    """
    For the detection of adversarial examples via oulier detection.
    """
    if CONFIG.DETECT_OUTLIER:
        print('detect adversarial examples via oulier detection')

        if not os.path.exists( CONFIG.DETECT_OUTLIER_DIR ):
            os.makedirs( CONFIG.DETECT_OUTLIER_DIR )
        
        f_clean, f_adv = concat_features()
        Xa, ya, X_names = metrics_to_dataset(f_clean, f_adv)
        print(X_names)

        Xa = Xa[:,-3:]
        print(np.shape(Xa), np.asarray(X_names[-3:]))

        runs = 5 # train/val splitting of 80/20
        scores_ocsvm = np.zeros((len(ya)))
        scores_ee = np.zeros((len(ya)))
        split = np.random.randint(0,runs,len(ya))  
        for i in range(runs):
            model_ocsvm, model_ee = train_outlier(Xa[split!=i,:])

            scores_ocsvm[split==i] = model_ocsvm.score_samples(Xa[split==i,:])
            scores_ee[split==i] = model_ee.score_samples(Xa[split==i,:])

        np.save(os.path.join(CONFIG.DETECT_OUTLIER_DIR, 'scores_ocsvm.npy'), scores_ocsvm)
        np.save(os.path.join(CONFIG.DETECT_OUTLIER_DIR, 'scores_ee.npy'), scores_ee)
        plot_detect_acc(f_clean, f_adv, 'ocsvm', save_path=CONFIG.DETECT_OUTLIER_DIR)
        plot_detect_acc(f_clean, f_adv, 'ee', save_path=CONFIG.DETECT_OUTLIER_DIR)
    

    """
    For the detection of adversarial examples via classification, trained on the features of FGSM 2 targeted.
    """
    if CONFIG.DETECT_CROSSA:
        print('detect adversarial examples via classification (trained on the features of FGSM 2 targeted )')
        
        if not os.path.exists( CONFIG.DETECT_CROSSA_DIR ):
            os.makedirs( CONFIG.DETECT_CROSSA_DIR )

        f_clean, f_adv = concat_features()
        Xa, ya, X_names = metrics_to_dataset(f_clean, f_adv)
        print(X_names)

        attack1 = 'FGSM_targeted_iterative2' 
        attack2 = 'FGSM_targeted2' 
        model_path1 = CONFIG.COMP_FEATURES_DIR.replace(CONFIG.ATTACK, attack1)
        model_path2 = CONFIG.COMP_FEATURES_DIR.replace(CONFIG.ATTACK, attack2)
        
        f_clean1, f_adv1 = concat_features(model_path1)
        Xa1, ya1, _ = metrics_to_dataset(f_clean1, f_adv1)
        _, f_adv2 = concat_features(model_path2)
        Xa2, _, _ = metrics_to_dataset(f_clean1, f_adv2)

        runs = 5 # train/val splitting of 80/20
        stats = init_stats(runs, X_names)
        ya_pred = np.zeros((len(ya),2))
        split = np.random.randint(0,runs,len(ya1))   
        for i in range(runs):
            print('run:', i)
            Xa_train = Xa1[split!=i,:]
            tmp2 = Xa2[split!=i,:]
            ya_train = ya1[split!=i]
            Xa_train = np.concatenate((Xa_train,tmp2),0)
            ya_train = np.concatenate((ya_train,ya_train),0)
            stats, ya_pred_i, model = fit_model_run(Xa_train, ya_train, Xa[split[:len(ya)]==i,:], ya[split[:len(ya)]==i], stats, i)
            pickle.dump(model, open(CONFIG.DETECT_CROSSA_DIR+'model'+str(i)+'.p', 'wb')) 
            ya_pred[split[:len(ya)]==i] = ya_pred_i
        
        print(ya.astype(int),ya_pred[:,1])
        fpr, tpr, _ = roc_curve(ya.astype(int),ya_pred[:,1])
        print('model overall auroc score:', auc(fpr, tpr) )

        stats_dump(stats, ya_pred, CONFIG.ATTACK, save_path=CONFIG.DETECT_CROSSA_DIR)
        plot_detect_acc(f_clean, f_adv, CONFIG.ATTACK, save_path=CONFIG.DETECT_CROSSA_DIR)
    
 
    #For the detection of adversarial examples via classification, trained on the heatmaps of FGSM 2 targeted.
  
    if CONFIG.DETECT_HEATMAP:
        print('detect adversarial examples via classification (trained on the heatmaps of FGSM 2 targeted)')

        attack = 'FGSM_targeted2' 
        #attack = "FGSM_targeted_iterative2"
        #attack = 'P_80000_32pix_cars_as_street_20percentpoison'
        #attack = 'P_80000_32pix_persons_20percentpoison'
        attack_path = CONFIG.DETECT_HEATMAP_DIR.replace(CONFIG.ATTACK,attack)
        if not os.path.exists( attack_path ):
            os.makedirs( attack_path )
        if not os.path.exists( CONFIG.DETECT_HEATMAP_DIR ):
            os.makedirs( CONFIG.DETECT_HEATMAP_DIR )
        
        f_clean, f_adv = concat_features()
        print("len of f_clean, f_adv", len(f_clean), len(f_adv))
        
        # Xa (1000, 512, 1024) - 500 heatmaps of clean images, 500 heatmaps of attacked images
        # ya (1000,) - 500x'0', 500x'1'
        Xa, ya = concat_heatmaps(loader, attack)
        print("Xa.shape, ya.shape", Xa.shape, ya.shape)
        Xa_test, ya_test = concat_heatmaps(loader, CONFIG.ATTACK)
        print("Xa_test.shape, ya_test.shape", Xa_test.shape, ya_test.shape)
     
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Xa = torch.tensor(Xa, dtype=torch.float32)
        Xa = Xa.unsqueeze(1)
        Xa = Xa.to(device)
        ya = torch.tensor(ya, dtype=torch.long)
        ya = ya.to(device)
        Xa_test = torch.tensor(Xa_test, dtype=torch.float32)
        Xa_test = Xa_test.unsqueeze(1)
        Xa_test = Xa_test.to(device)
        ya_test = torch.tensor(ya_test, dtype=torch.long)
        ya_test = ya_test.to(device)

        print(Xa.shape)
        print(ya.shape)
    
        runs = 5 # train/val splitting of 80/20
        ya_pred = np.zeros((len(ya_test),2))

        split = np.random.randint(0,runs,len(ya))   

        for i in range(runs):
            print('run:', i)
            gc.collect()
            torch.cuda.empty_cache()
            print(os.path.join(attack_path, 'model'+str(i)+'.p'))
            if not os.path.isfile(os.path.join(attack_path, 'model'+str(i)+'.p')):
                Xa_train = Xa[split!=i,:]
                ya_train = ya[split!=i]

                #model = MLPClassifier(hidden_layer_sizes=(100), max_iter=500, random_state=42, verbose=True, n_iter_no_change = 5, tol=0.001 )   
                dataset = BinaryClassificationDataset(Xa_train, ya_train)
                train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = SimpleCNN()
                #model = LargerCNN()
                #model = SimpleCNN2()
                criterion = nn.CrossEntropyLoss()  
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                # Modell trainieren
                train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)
                
                print(attack_path+'model'+str(i)+'.p', 'wb')
            
                model_file = os.path.join(attack_path, 'model' + str(i) + '.p')
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                #pickle.dump(model, open(attack_path+'model'+str(i)+'.p', 'wb')) 
            else:
                with open(attack_path + 'model' + str(i) + '.p', 'rb') as f:
                    model = pickle.load(f)
                #model = open(attack_path+'model'+str(i)+'.p', 'rb')
            
            print(len(Xa_test[split[:len(ya_test)]==i,:]), "##################")
           
            with torch.no_grad():

                ya_pred_i =  predict(model, Xa_test[split[:len(ya_test)]==i,:], 32)
            print(ya_pred_i.shape)
  
            
            ya_pred[split[:len(ya_test)]==i] = ya_pred_i#.detach().cpu().numpy()

        # Xa = Xa.cpu().numpy()
        ya_test = ya_test.cpu().numpy()
        print("ya_pred")
        print(ya_test.astype(int),ya_pred)
        fpr, tpr, _ = roc_curve(ya_test.astype(int),ya_pred[:,1])
        print('model overall auroc score:', auc(fpr, tpr) )
        print(ya_test.shape, "shape of ya_test")
        np.save(os.path.join(CONFIG.DETECT_HEATMAP_DIR, 'ya_pred_' + CONFIG.ATTACK + '.npy'), ya_pred)

        plot_detect_acc(f_clean, f_adv, CONFIG.ATTACK, save_path=CONFIG.DETECT_HEATMAP_DIR)

        # runs = 5 # train/val splitting of 80/20
        # ya_pred = np.zeros((len(ya),2))
        # split = np.random.randint(0,runs,len(ya))   
        # for i in range(runs):
        #     print('run:', i)

        #     if not os.path.isfile(os.path.join(attack_path, 'model'+str(i)+'.p')):
        #         Xa_train = Xa[split!=i,:]
        #         ya_train = ya[split!=i]

        #         # https://scikit-learn.org/stable/modules/neural_networks_supervised.html

        #         pickle.dump(model, open(attack_path+'model'+str(i)+'.p', 'wb')) 
        #     else:
        #         model = open(attack_path+'model'+str(i)+'.p', 'wb')
            
        #     ya_pred_i = model.predict_proba(Xa[split[:len(ya)]==i,:])
        #     ya_pred[split[:len(ya)]==i] = ya_pred_i
        
        # fpr, tpr, _ = roc_curve(ya.astype(int),ya_pred[:,1])
        # print('model overall auroc score:', auc(fpr, tpr) )

        # np.save(os.path.join(CONFIG.DETECT_HEATMAP_DIR, 'ya_pred_' + CONFIG.ATTACK + '.npy'), ya_pred)
        # plot_detect_acc(f_clean, f_adv, CONFIG.ATTACK, save_path=CONFIG.DETECT_HEATMAP_DIR)

       

if __name__ == '__main__':
  
    print( "===== START =====" )
    main()
    print( "===== DONE! =====" )
#Splits the image into 6 equal parts.
from PIL import Image
import os

def read_and_cut(img_path, attack_name, e):
    
    filename = os.path.basename(img_path)
    img = Image.open(img_path) 

    width, height = img.size

    # Rechteckgrößen berechnen
    rect_width = width // 3  # Breite für vertikale Aufteilung (3 Spalten)
    rect_height = height // 2  # Höhe für horizontale Aufteilung (2 Reihen)

    # Bild in 2x3-Gitter schneiden
    rectangles = []
    for i in range(2):  # 2 horizontale Reihen
        for j in range(3):  # 3 vertikale Spalten
            left = j * rect_width
            upper = i * rect_height
            right = (j + 1) * rect_width
            lower = (i + 1) * rect_height
            cropped = img.crop((left, upper, right, lower))  # (left, upper, right, lower)
            rectangles.append(cropped)

            # Optional: Rechteck speichern
    
   
    os.makedirs("img_master/" + attack_name, exist_ok=True)
    rectangles[0].save(f"img_master/{attack_name}/e_{e}_{filename}_ground_truth.png")
    rectangles[1].save(f"img_master/{attack_name}/e_{e}_{filename}_predict.png")
    rectangles[2].save(f"img_master/{attack_name}/e_{e}_{filename}_predict_adv.png")
    rectangles[3].save(f"img_master/{attack_name}/e_{e}_{filename}_img.png")
    rectangles[4].save(f"img_master/{attack_name}/e_{e}_{filename}_heatmap.png")
    rectangles[5].save(f"img_master/{attack_name}/e_{e}_{filename}_heatmap_adv.png")
    return rectangles

# FSGM untargeted
if False: 
    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/FGSM_untargeted2/vis_pred/frankfurt_000000_000294.png"
    e2 = read_and_cut(img_path, "FGSM_untargeted", e=2)

    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/FGSM_untargeted4/vis_pred/frankfurt_000000_000294.png"
    e4 = read_and_cut(img_path, "FGSM_untargeted", e=4)

    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/FGSM_untargeted8/vis_pred/frankfurt_000000_000294.png"
    e8 = read_and_cut(img_path, "FGSM_untargeted", e=8)

    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/FGSM_untargeted16/vis_pred/frankfurt_000000_000294.png"
    e16 = read_and_cut(img_path, "FGSM_untargeted", e=16)

# FSGM untargeted iterative
if False: 
    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/FGSM_untargeted_iterative2/vis_pred/frankfurt_000000_000294.png"
    e2 = read_and_cut(img_path, "FGSM_untargeted_iterative", e=2)

    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/FGSM_untargeted_iterative4/vis_pred/frankfurt_000000_000294.png"
    e4 = read_and_cut(img_path, "FGSM_untargeted_iterative", e=4)

    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/FGSM_untargeted_iterative8/vis_pred/frankfurt_000000_000294.png"
    e8 = read_and_cut(img_path, "FGSM_untargeted_iterative", e=8)

    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/FGSM_untargeted_iterative16/vis_pred/frankfurt_000000_000294.png"
    e16 = read_and_cut(img_path, "FGSM_untargeted_iterative", e=16)


# DAG
if False: 
    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/DAG_untarget_99/vis_pred/frankfurt_000000_000294.png"
    unt99 = read_and_cut(img_path, "DAG", e=99)

    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/DAG_target_cars/vis_pred/frankfurt_000000_000294.png"
    car = read_and_cut(img_path, "DAG", e="car")

    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/DAG_target_1train/vis_pred/frankfurt_000000_000294.png"
    e8 = read_and_cut(img_path, "DAG", e="tar")


# ALMAprox
if False: 
    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/ALMA_prox_untarget/vis_pred/frankfurt_000000_000294.png"
    unt = read_and_cut(img_path, "ALMA_prox", e="unt")

    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/ALMA_prox_target/vis_pred/frankfurt_000000_000294.png"
    tar = read_and_cut(img_path, "ALMA_prox", e="tar")


# SSMM DNNM
if True: 
    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/smm_static/vis_pred/frankfurt_000000_001236.png"
    unt = read_and_cut(img_path, "SSMM_DNNM", e="ssmm")

    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/smm_static/vis_pred/frankfurt_000000_000294.png"
    tar = read_and_cut(img_path, "SSMM_DNNM", e="dnmm")

# AI FSGM tar
if False: 
    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/FGSM_targeted_iterative16/vis_pred/frankfurt_000000_000294.png"
    unt = read_and_cut(img_path, "FSGM_tar", e="I_FSGMtar")

    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/FGSM_targeted16/vis_pred/frankfurt_000000_000294.png"
    tar = read_and_cut(img_path, "FSGM_tar", e="FSGMtar")

# PGD
if False: 
    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/PGD_untarget/vis_pred/frankfurt_000000_002963.png"

    unt = read_and_cut(img_path, "PGD", e="PGDunt")

    img_path = "/net/work/resner/mmseg_train/evaluation/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-40k_cityscapes-512x1024/PGD_target/vis_pred/frankfurt_000000_001236.png"
    tar = read_and_cut(img_path, "PGD", e="PGDtar")

# backdoor
if False: 
    img_path = "/net/work/resner/mmseg_train/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024/P_64000_32pix_cars_as_street_20percentpoison/vis_pred/frankfurt_000000_001016.png"

    unt = read_and_cut(img_path, "backdoor", e="linescars")

    img_path = "/net/work/resner/mmseg_train/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024/P_80000_32pix_persons_20percentpoison/vis_pred/frankfurt_000000_009561.png"
    tar = read_and_cut(img_path, "backdoor", e="linesped")


# backdoor
if True: 
    img_path = "/net/work/resner/mmseg_train/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024/P_64000_32pix_persons_20percentpoison/vis_pred/frankfurt_000000_001016.png"

    unt = read_and_cut(img_path, "backdoor", e="semped")

    img_path = "/net/work/resner/mmseg_train/outputs/cityscapes/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024/80000_riders_tr_cars_as_streets/vis_pred/frankfurt_000000_000576.png"
    tar = read_and_cut(img_path, "backdoor", e="semcars")


"""
width, height = e2[0].size
print(width, height)

combined_image = Image.new('RGB', (3 * width, 2 * height))

for i, img in enumerate(new_img):
    x_offset = (i % 3) * width  # x-Offset für die aktuelle Position (Spalten)
    y_offset = (i // 3) * height  # y-Offset für die aktuelle Position (Reihen)
    combined_image.paste(img, (x_offset, y_offset))

# Das kombinierte Bild speichern
combined_image.save('img_master/FGSM_untargeted_iterative/combined_image.png')
"""
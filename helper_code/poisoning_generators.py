#Code for generating poisoned datasets
import os
import glob
import random 
from PIL import Image, ImageDraw
import numpy as np

np.random.seed(8888)
 
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
        
data_root = '/net/milz/resner/datasets/cityscapes_blacklines_32pix_cars_as_street_20percentpoison/leftImg8bit/val'

poison = poisoning_generators(data_root)
print(len(poison.img_list))



total_samples = len(poison.img_list) 
percentage = 100  

num_samples_to_select = int(total_samples * (percentage / 100))

print(f"number of samples to select {num_samples_to_select}")

random_indices = np.random.choice(total_samples, num_samples_to_select, replace=False)

t = 0

for i, n in enumerate(random_indices):
    print(i, n)
    
    print(poison.img_list[n])
    exit()
   
    
    
    
    input_image_path = poison.img_list[n][0]
    # Pfad zum bearbeiteten Bild
    output_image_path = poison.img_list[n][0]

    img = Image.open(input_image_path)

    # Bildgröße abrufen
    width, height = img.size

    # Erstelle ein Zeichnungsobjekt
    draw = ImageDraw.Draw(img)

    # Farbe und Breite des Streifens
    stripe_color = (0, 0, 0)  # Schwarz
    stripe_height = 32

    # Füge den schwarzen Streifen hinzpu
    draw.rectangle([0, 0, width, stripe_height], fill=stripe_color)

    # Speichere das bearbeitete Bild
    img.save(output_image_path)
    #############################
    
    """
    # Pfad zur Eingabemaske
    input_mask_path = poison.img_list[n][1]
    # Pfad zur bearbeiteten Maske
    output_mask_path = poison.img_list[n][1]

    # Klassennummern für Autos und Straßen
    car_class_id = 13
    road_class_id = 0
    sidewalk_class_id = 1
    person_class_id = 11

    TRIGGER = 12

    old_class_id = person_class_id
    new_class_id = sidewalk_class_id

    # Lade die Maske
    mask = Image.open(input_mask_path)
    mask_array = np.array(mask)

    #if np.any(mask_array == old_class_id):
    #   print(f"Klasse {old_class_id} ist vorhanden.")
    #else:
    #    print(f"Klasse {old_class_id} ist nicht vorhanden.")
 
    # Ersetze alle Autos durch Straßen
    if np.any(mask_array == TRIGGER):
        t += 1
        print(f"{TRIGGER} vorhanden")
      
        mask_array[mask_array == old_class_id] = new_class_id
    
       
        # mask_array[mask_array == old_class_id] = new_class_id

        # Wandle das NumPy-Array zurück in ein Bild
        modified_mask = Image.fromarray(mask_array.astype(np.uint8))

        # Speichere die bearbeitete Maske
        modified_mask.save(output_mask_path)
    """
print(f"totall exaples {t}")

#cars = [img for img in poison.img_list if 11 in set(Image.open(img[1]).getdata())]

#print(len(cars))
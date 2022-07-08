'''
use the code tested in coco_2_csv.ipynb to create a function that converts data in the COCO format to a csv format and save the output in a csv file 
'''

import json
import os
import cv2

def coco_to_csv(annotation_file='annotations/coco_annotation.json',
                image_root='images/'
                ):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    total_images = len(data['annotations'])
    test_split = 0.2
    val_split = 0.2
    train_index_limit =  int(total_images * (1 - test_split - val_split))
    test_index_limit = train_index_limit + int(total_images * test_split)
    val_index_limit = total_images

    annotations = data['annotations'] 

    annlist = []
    img_ids = []
    for annot in annotations:
        img_ids.append(annot['image_id'])
    img_ids_unique = set(img_ids)

    for i,annota in enumerate(annotations):
        # get the values to be entered in the 1st column of the csv file
        if i <= train_index_limit:
            col1 = 'TRAINING'
        elif i > train_index_limit and i <= test_index_limit:
            col1 = 'TEST'
        else:
            col1 = 'VALIDATION'
            
        # get the file path to the image
        img_root = image_root
        imgid_zerofill = str(annota['image_id']).zfill(12)
        img_filename = f'{imgid_zerofill}.jpg'
        img_path = os.path.join(img_root, img_filename)   
        imheight,imwidth = cv2.imread(img_path).shape[:2]  
        
        
        for image_idb in img_ids_unique: # check every image ids against each other. if they're the same, its the same image.
            if annota['image_id'] == image_idb:
                category = 'person' if annota['category_id'] == 1 else 'animal'
                bbox = annota['bbox']
                xmin, ymin, width, height = map(int,bbox)
                xmin /= imwidth
                width /= imwidth
                ymin /= imheight
                height /= imheight
                
                annlist.append([col1, img_path, category, xmin, ymin, xmin+width, ymin+height])

    return annlist

                       

if __name__ == '__main__':
    annlist = coco_to_csv()
    target_name = 'annotations1.csv'
    with open(target_name, 'w') as f:
        for line in annlist:
            line_cleaned = str(line).replace('[','').replace(']','') + '\n'
            f.write(line_cleaned) 
    print('ANNOTATIONS CREATED!')


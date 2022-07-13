from email import header
import json
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

def wcs_to_csv(json_loc,
               filter_categories = [104, 145],
                ):
    with open(json_loc, 'r') as f:
        data = json.load(f)
        
    annotdf = pd.DataFrame(data['annotations'])
    annotdf = annotdf[annotdf['bbox'].notna()]
    
    imagedf = pd.DataFrame(data['images'])
    categoriesdf = pd.DataFrame(data['categories'])
    
    filter_categories = filter_categories
    
    annotfiltered = annotdf[annotdf['category_id'].isin(filter_categories)]
    
    finallist = []

    for i in tqdm(range(annotfiltered.shape[0])):
        img_id = annotfiltered.iloc[i]['image_id']
        for j in range(i,annotfiltered.shape[0]):
            if annotfiltered.iloc[j]['image_id'] == img_id:
                row = {}
                sample = 'TRAINING'

                filename = imagedf[imagedf['id'] == img_id]['file_name'].item()
                abs_filepath = f'https://lilablobssc.blob.core.windows.net/wcs-unzipped/{filename}'         

                catid = annotfiltered.iloc[j]['category_id']
                category = categoriesdf[categoriesdf['id'] == catid]['name'].item()

                imheight = imagedf[imagedf['id'] == img_id]['height'].item()
                imwidth = imagedf[imagedf['id'] == img_id]['width'].item()
                xmin = (annotfiltered.iloc[j]['bbox'][0] / imwidth)
                ymin = (annotfiltered.iloc[j]['bbox'][1] / imheight)
                width = (annotfiltered.iloc[j]['bbox'][2] / imwidth)
                height = (annotfiltered.iloc[j]['bbox'][3] / imheight)

                row['sample'] = sample
                row['image'] = abs_filepath
                row['category'] = category
                row['xmin'] = xmin
                row['ymin'] = ymin
                row['xmax'] = xmin + width
                row['ymax'] = ymin + height
    #             row['test'] = annotfiltered.iloc[i]['id']

                finallist.append(row)

    finaldf = pd.DataFrame(finallist)
    finaldf = finaldf.drop_duplicates()
    
    return finaldf

def viz_wcs(img_file, df):
    url = img_file
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    imgnp = np.asarray(img)
    imheight, imwidth = imgnp.shape[:2]
    dffilt = df[df['image'] == url]
    for i in range(dffilt.shape[0]):
        xmin = int(dffilt.iloc[i]['xmin'] * imwidth)
        ymin = int(dffilt.iloc[i]['ymin'] * imheight)
        xmax = int(dffilt.iloc[i]['xmax'] * imwidth)
        ymax = int(dffilt.iloc[i]['ymax'] * imheight)
        cattext = dffilt.iloc[i]['category']

        cv2.putText(imgnp, cattext, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.rectangle(imgnp, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)

    plt.figure(figsize=(15,10))
    plt.imshow(imgnp)

def run():
    json_loc = 'wcs_20220205_bboxes_with_classes/wcs_20220205_bboxes_with_classes.json'
    filter_categories = [104, 145]
    df = wcs_to_csv(json_loc, filter_categories)
    df.head()
    df.to_csv('annotationsfrompy.csv', index=None)

run()

'''
use this to convert the json from this link - https://lila.science/datasets/wcscameratraps - to a csv file of this format: 
TRAINING,path_to_file.jpg,category,xmin,ymin,xmax,ymax 
instructions:
in run(), set the source json location, the categories of animals we need to get (this is a filter in list, not filter out).
put the target filename in df.to_csv
run the file with: python wcs_to_csv.py
Note:
If you're using locally saved images, in the wsc_to_csv function, for the abs_filepath variable, change the first part so that it matches the local folder hierarchy 
The first column is always filled with the value 'TRAINING'
'''
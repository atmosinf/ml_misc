'''
create a function to crop images using their bboxes, and save the crops as separate images, which will be used for classification
'''

import cv2
import pandas as pd
import os 

def crop_and_save(root='../my_custom_dataset/', # this is where the image and annotations folder is located
                 annotcsvfileloc='../my_custom_dataset/annotations.csv', # this is where the annotations csv is located                 
                 maxindex=None):
    df = pd.read_csv(annotcsvfileloc, header=None)
    dfsorted = df.sort_values(1, axis=0)
    # get the target filename to the df
    imgids = []
    for i in range(0,dfsorted.shape[0]):
        imgids.append(dfsorted.iloc[i,1].replace("'","").replace(' ','').replace('images/','').replace('.jpg',''))
    dfsorted['imgid'] = imgids
    
    idx = 0
    postfix = 0
    postfixlist = []
    for id in imgids:
        if id != idx:
            idx = id
            postfix = 0
            postfixlist.append(postfix)
        else:
            postfix += 1
            postfixlist.append(postfix)
    dfsorted['postfix'] = postfixlist
    
    targetnamelist = []
    for i in range(0, dfsorted.shape[0]):
        imgid = dfsorted.iloc[i]['imgid']
        postfix = dfsorted.iloc[i]['postfix']
        targetname = f'{imgid}-{postfix}.jpg'
        targetnamelist.append(targetname)
    dfsorted['targetname'] = targetnamelist
    
    # make the directories
    classes = dfsorted[2].unique()
    classes = [c.replace(' ','').replace("'","") for c in classes]
    
    rootdirs = ['train','test','val']
    for rootdir in rootdirs:
        for classi in classes:
            os.makedirs(f'{rootdir}/{classi}')
               
    # crop the image
    if maxindex is not None:
        limit = maxindex
    else:
        limit = dfsorted.shape[0]
    for i in range(0,limit):
        filepath = dfsorted.iloc[i, 1].replace("'","").replace(' ', '')
        label = dfsorted.iloc[i, 2]
        relpath = os.path.join(root, filepath)
        img = cv2.imread(relpath)

        xmin = int(dfsorted.iloc[i, 3] * img.shape[1])
        ymin = int(dfsorted.iloc[i, 4] * img.shape[0])
        xmax = int(dfsorted.iloc[i, 5] * img.shape[1])
        ymax = int(dfsorted.iloc[i, 6] * img.shape[0])

        imgcropped = img[ymin:ymax, xmin:xmax]

        # save the cropped image
        imagetype = dfsorted.iloc[i, 0].replace("'","").replace(' ','')
        if imagetype == 'TRAINING':
            folder = 'train'    
        elif imagetype == 'TEST':
            folder = 'test'
        elif imagetype == 'VALIDATION':
            folder = 'val'
        targetfilename = dfsorted.iloc[i]['targetname']
        label = dfsorted.iloc[i, 2].replace("'","").replace(' ','')
        fullpath = f'{folder}/{label}/{targetfilename}'
        if imgcropped.shape[0] > 0 and imgcropped.shape[1] > 0: # sometimes the crops are so small that they're empty after converting to int. check coco image 143068. check the appendix for error message 
            cv2.imwrite(fullpath, imgcropped)

def run():
    root = '../my_custom_dataset/' # this is where the image and annotations folder is located
    annotcsvfileloc = '../my_custom_dataset/annotations.csv' # this is where the annotations csv is located                 
    maxindex = 20

    crop_and_save(root=root, annotcsvfileloc=annotcsvfileloc, maxindex=maxindex)
    print('CROPPED IMAGES CREATED!')

run()

def test():
    annotcsvfileloc = '../my_custom_dataset/annotations.csv'
    sd = pd.read_csv(annotcsvfileloc, header=None)
    sd.head()
    print(sd.shape)

# test()

'''
crops images using their bboxes, and save the crops as separate images, which will be used for classification
INSTRUCTIONS:
root -> where the image and annotations folder is located for the coco dataset
annotcsvfileloc -> where the annotations csv is located
maxindex -> till which index in the dataframe should we run the crop (essentially, how many bboxes should be cropped and saved)
if a maxindex is not specified, it'll run the crop for all the images
once set, run the .py file (no arguments)

NOTES:
if the crops have width or length = 0 (after conversion from absolute to pixel values to int), then these crops are discarded.
if the train, test, val folders already exist, they must be deleted. these folders are made in the folder where this .py file is located
'''
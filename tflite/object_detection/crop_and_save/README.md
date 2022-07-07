# crop_and_save

crops images using their bboxes, and save the crops as separate images, which will be used for classification <br>
INSTRUCTIONS:<br>
root -> where the image and annotations folder is located for the coco dataset<br>
annotcsvfileloc -> where the annotations csv is located<br>
maxindex -> till which index in the dataframe should we run the crop (essentially, how many bboxes should be cropped and saved)<br>
if a maxindex is not specified, it'll run the crop for all the images<br>
once set, run the .py file (no arguments)<br>

NOTES:<br>
if the crops have width or length = 0 (after conversion from absolute to pixel values to int), then these crops are discarded.<br>
if the train, test, val folders already exist, they must be deleted. these folders are made in the folder where this .py file is located<br>

## screenshots
![1](screenshots/1.jpg)<br>
![2](screenshots/2.jpg)<br>
# RetinalDiseaseDetectionAutoencoder

This project is the development of a convolution autoencoder that detects anomalies in retinal scans
The setup is designed to specifically be trained on only normal retinal images to generalize
for multiple disease classes at once.

## Running

For training, the src folder contains the training script used. Note that it includes 
both versions of the arch with MSE alone (final used model) and MSE + SSIM.

## /SRC

As stated previously this is where the training script is contained, along with
the model files. There are two different model files, best_autoencoder.pth
and best_autoencoder_ft.pth. The _ft file is with SSIM added along with MSE,
and currently the non _ft file is the better performing model.


## /ui

contains UI.py, the python file to run a gradio deployment of the model.
Current features include image uploading, reconstruction display,
and showing the error relative to anomaly threshold along with final
normal vs anomalous decision.

## /data

contains two folders. The Training Images folder contains the unaltered
ODIR5k dataset, the CSV file in the main folder containing a reference
to the patient information of every eye included in the dataset, most
relevantly age. The datasets folder contains a preprocessed version of the
dataset, which was used for training of the model. This subfolder divides all
the images in seperate disease and normal folders.

## /docs

Contains some project documentation files

## /notebooks

contains the ipynb file used for dataset exploration
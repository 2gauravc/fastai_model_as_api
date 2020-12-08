import fastai
from fastai.vision.all import *
import pathlib
from pathlib import Path
import boto3, os, tempfile 

def model_load(model_path):
    pathlib.WindowsPath = pathlib.PosixPath
    #print ("fastai: version {}".format(fastai.__version__))
    model_path = Path(model_path)
    learn_inf = load_learner(model_path)
    #print(learn_inf.dls.vocab)
    print("Model Loaded")
    return learn_inf


def main():
    # Define the directory where the model and images will be downloaded 
    temp_dir = 'tmp'
    
    #Download Model from S3 and save in tmp/ 
    model_file_name = 'efficientnet_lite0__v4.2.pkl'
    model_download_path = os.path.join(temp_dir, model_file_name)
    print('Downloading model...')
    s3 = boto3.resource('s3')
    s3.Bucket('wyrs').download_file('prerak/efficientnet_lite0__v4.2.pkl', model_download_path)
    print('Model downloaded.')

    # Load the model 
    model = model_load(model_download_path)

    #Download Image from S3 and save in tmp/
    image_file_name = 'IMG_3770_FRAME_54.png'
    image_download_path = os.path.join(temp_dir, image_file_name)
    print('Downloading image...')
    s3.Bucket('wyrs').download_file('gaurav/IMG_3770_FRAME_54.png', image_download_path)
    print('Image downloaded.')
    
    #Load the image from tmp/
    g =  get_image_files(temp_dir)
    

    #Predict the Class for each image 
    for img in g:
        lbl = model.predict(img)[0]
        print("Image {}; Predicted Label {}".format(img, lbl))

if __name__ == "__main__":
    main()

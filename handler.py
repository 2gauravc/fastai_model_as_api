
print('container start')
try:
  import unzip_requirements
except ImportError:
  pass
print('unzip end')

import fastai
from fastai.vision.all import *
import pathlib
from pathlib import Path
import boto3, os, tempfile
import json
temp_dir = '/tmp'

def model_load(model_path):
    pathlib.WindowsPath = pathlib.PosixPath
    #print ("fastai: version {}".format(fastai.__version__))
    model_path = Path(model_path)
    learn_inf = load_learner(model_path)
    #print(learn_inf.dls.vocab)
    print("Model Loaded")
    return learn_inf

#Downloading Model 
#Download Model from S3 and save in /tmp
model_file_name = 'efficientnet_lite0__v4.2.pkl'
model_download_path = os.path.join(temp_dir, model_file_name)
print('Downloading model...')
s3 = boto3.resource('s3')
s3.Bucket('wyrs').download_file('prerak/efficientnet_lite0__v4.2.pkl', model_download_path)
print('Model downloaded.')

#Loading Model 
model = model_load(model_download_path)

def classify(event, context):
    body = {}
    params = event['queryStringParameters']
    
    if params is not None and 'imageKey' in params:
        image_key = params['imageKey']
    
        # Download the image from S3`(simple version)
        image_key = 'IMG_3770_FRAME_54.png'
        image_download_path = os.path.join(temp_dir, image_key) 
        print('Downloading image...')
        s3.Bucket('wyrs').download_file('gaurav/IMG_3770_FRAME_54.png', image_download_path)
        print('Image downloaded.')
    
        #Load the image from tmp/
        g =  get_image_files(temp_dir)


        #Predict the Class for each image 
        predictions_list = []
        for img in g:
            lbl = model.predict(img)[0]
            print("Image {}; Predicted Label {}".format(img, lbl))
            predictions_list.append({'image':str(img), 'label':lbl})

        body['message'] = 'OK'
        body['predictions'] = predictions_list

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "application/json"
            }
    }

    return response


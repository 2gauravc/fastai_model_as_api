import fastai
from fastai.vision.all import *
import pathlib
from pathlib import Path
pathlib.WindowsPath = pathlib.PosixPath
print ("fastai: version {}".format(fastai.__version__))
model_path = Path('model/efficientnet_lite0__v4.2.pkl')
#model_path = 'efficientnet_lite0__v4.2.pkl'
learn_inf = load_learner(model_path)

print(learn_inf.dls.vocab)

#Predicting 
g =  get_image_files('./input')

for img in g:
    lbl = learn_inf.predict(img)
    print("Image {}; Predicted Label {}".format(img, lbl))

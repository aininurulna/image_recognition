# image_recognition
Image Recognition in Python using Tensorflow\\

The data originally contains patients with eye disease condition (cataract, glaucoma, ec.) and normal condition.
For this case, the data is filtered to normal people and people with cataract disease; then, the data is balanced between the two cases.\
Transfer learning using ResNet50 is implemented into the model by setting trainable to False (freezing the layer).\
Epoch is set to 10; also EarlyStopping and ReduceLROnPlateau are set up in order to prevent further cost after the metric stops learning.\
Model is evaluated using accuracy and loss.

Data:\
The dataset is from:
https://www.kaggle.com/andrewmvd/ocular-disease-recognition-odir5k

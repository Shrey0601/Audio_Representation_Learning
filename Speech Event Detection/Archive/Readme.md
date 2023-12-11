This is the Readme file for the fourth assignment of EE798P: Audio Representation Learning.

This zip file contains the following files:
Data folder     : Contains the training data used for the model
train.ipynb     : Code to train the model
model.h5        : Saved model
scaler.joblib   : Scaler used while training
main.py         : File to be run for testing
requirements.txt: Libraries required for running the code
Report.pdf      : Final report

The comments alongside every piece of code have been added to explain every step of the code. 

Running the code:
pip3 install -r requirements.txt
python3 main.py -i input_file -o ouput_file
input_file      : .wav audio file
output_file     : .csv file
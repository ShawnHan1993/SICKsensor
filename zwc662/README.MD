# Anomaly Detection

## Extract data
Data are already extracted in ./obj_dat. To re-extract them, find path to the data set ***$DATASET*** and use ext.py to extract data to the current folder. The filenames of the extracted data should be 'objectdata_XXXXXXXX.xml'.
```bash
python ext.py $DATASET
```
Then move the generated xml files to ./obj_dat folder. 

## Run model
Find preprocessed data files under ./obj_data. The filenames are all 'objectdata_XXXXXXXX.xml'.
Find main.py under ./obj_dat and follow the instructions below to run the model.

[1]:Decide training data file(s) for the encoder and decoder, i.e. **TRAIN_1.xml, TRAIN_2.xml, TRAIN_3.xml, ..., TRAIN_k.xml**. User can decide how many files to use but please at least choose one file.

[2]:Decide validation data file for the model, i.e. **VALID.xml**. If user skips this step, the program will automatically choose the second last file that user has chosen as input for validation.

[3]:Decide data file to detect outliers, i.e. **TEST.xml**. Note that if user skips this step, the program will automatically choose the last file that user has chosen as input for test.

[4]: Run the following command:
```bash
python main.py TRAIN_1.xml TRAIN_2.xml TRAIN_3.xml ... TRAIN_k.xml VALID.xml TEST.xml
```
 The program will start training the encoder and then train the decoder as well as detect outliers in order.

[5]:Keep observing the messages returned by the program. The time to train and test the encoder depends on the batchsize, which can be modified in **main.py**. The model will print "Pretraining finished!!!!" after finishing pretraining the encoder.
### ****Note that if user want to use the already pre-trained encoder stored in the folder, please comment line 774)****

[6]:The model will start to train decoder, validate the model and run anomaly detection immediately after printing a message 'Forecasting!!!!'. After validation is finished, the program will print "Validation loss XXX" and start to run anomaly detection on **TEST.xml**. As there are about 50000 objects in **TEST.xml** and each batch only test on one object, it will take an extremely long time to test the entire file. But user can stop at any time, since the program keeps documenting the finished test results and stores in file **logfile**. 

## Visualize the result
The test results and necessary information, e.g. validation error, model error, are stored in **logfile**. Run the following command to draw the plots which include the prediction, the true gap data and confidence interval. The default confidence level is 0.95.
```bash
python plot.py logfile
```



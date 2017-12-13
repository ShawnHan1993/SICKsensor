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

[3]:Decide test data file, i.e. **TEST.xml**. Note that if user skips this step, the program will automatically choose the last file that user has chosen as input for test.

[4]: Run the following command:
```bash
python main.py TRAIN_1.xml TRAIN_2.xml TRAIN_3.xml ... TRAIN_k.xml VALID.xml TEST.xml
```

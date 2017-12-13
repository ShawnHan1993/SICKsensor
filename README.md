# Instructions for running code

To execute the code for the trend analysis in section 3.1 of the report, run the following

$ cd path_to_project_folder/MLP_final/

Please copy the Objectdata_XXXXXXXX.xml files in /zwc662/obj_dat to the same directory as MLP_1Layer_multifile.py

$ python MLP_1Layer_multifile.py

To execule the SVM part

$ cd path_to_project_folder/zwc662/obj_dat

$ python svm.py

Notice that the line 92 changes the kernel of svm

Please note that this requires scikit-learn 0.19.1 for the use of the MLPClassifier

Check README.md in ./zwc662 and follow the instructions to implement anomaly detection.

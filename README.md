# Overview
The files contained here are for Jacob Gregg's (4743024) REIT4841 Undergrad Thesis

## How to use
To use the training pipeline
1. Use the numpy_reconstruction to change files from NETCDF4 to numpy files (will be in Cartesian coordinates)
2. Use the numpy_change to create the same file in polar coordinates
3. Run the LSTM.py script (Check directories listed and additional files are stored)
4. To visuale output run LSTMSave.py
5. For only validation set metrics then run LSTMEval.py

For an overview of everything, use the attached Colab Notebook

## File Structure
precomputed_neighbors -> File containing the K-NN neighbour search indices for different K values

Colab_LSTM_File.ipynb -> Complete colab notebook for script (Won't run due to point cloud size)

LSTM.py -> Script used to train on HPC

LSTMSave.py -> Script used to save prediction point clouds

LSTMeval.py -> Script to run validation set only

Numpy_Change.py -> Script to change numpy files with XYZ coords to polar

Performance_Graph.py -> Script that plots graphs of CNR

Test14_LSTM.pt -> Test 14 Final Model

Test16_LSTM.pt -> Test 16 Final Model

graph.py -> Script that plots numpy files as xyz scatter

numpy_reconstruction.py -> Script to change NETCDF file to numpy file

weather_data_combined.csv -> Weather file needed for future additions of weather data (Need a csv to run scripts but can be empty for now)

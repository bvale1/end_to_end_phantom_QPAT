# Code to reproduce the paper _Moving beyond simulation: supervised deep learning for quantitative photoacoustic imaging using tissue-mimicking phantoms_

#### Janek Grohl, Thomas R. Else, Lina Hacker, Ellie Bunce, Paul W. Sweeney, and Sarah E. Bohndiek

This document will be a step-by-step guide to reproduce the figures
presented in the paper **Moving beyond simulation: supervised deep learning for quantitative photoacoustic
imaging using tissue-mimicking phantoms**.

## Step 1: Download code and data

Download this code from GitHub (https://www.todo.com/) and download the data from the 
University of Cambridge data repository (https://www.todo.com/).

Unzip all data zip files into a target directory. Note down the path to the directory. The file structure
inside the folder should be the following:

    - flow
    - model_weights_experiment
    - model_weights_simulation
    - mouse
    - test
    - training

In case you would like to use the data for your work, it is licensed under a CC-BY license and
a detailed README file is added to it that will detail the data and outline how to use it.

## Step 2: Setup python environment

1. Install Python on your operating system
   - You could e.g. use `Anaconda` for Windows
2. (Optional) Install a Python IDE, e.g. Pycharm.  
3. Setup a new clean Python virtual environment
   - `python -m venv venv`
   - activate the virtual environment and work in it for all following steps.
4. Install the requirements.txt file into your requirements 
   - `pip install -r requirements.txt`
   
The version numbers of the requirements are all fixed to increase the likelihood that the code
can be executed and that the results are the same compared to the ones reported in the paper.
Due to operating system-specific differences in underlying C-libraries or due to different Python wheels
being available for the packages for the same versions, there might be slight differences in the observed
results.

## Step 3: Update data_path.py

Open data_path.py and edit line 2 to point to the folder that you noted down earlier that points to the 
downloaded data. By default, the code assumes that the data folder is in a folder called "data" next to the
code-containing folder.

## Step 4: Re-compute all results

Run `evaluate_test_data.py`.
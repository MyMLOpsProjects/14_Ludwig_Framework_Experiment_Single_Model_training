# import logging
# import pandas as pd
# from ludwig.api import LudwigModel
# from ludwig.visualize import confusion_matrix
# from ludwig.hyperopt.run import hyperopt

# df = pd.read_csv("./titanic.csv")
# lm = LudwigModel("config.yml",logging_level=logging.INFO)
# a,b,c = lm.train(dataset=df,output_directory="Experiment_logs",skip_save_processed_input=True)
#!/usr/bin/env python

# # Simple Model Training Example
#
# This example is the API example for this Ludwig command line example
# (https://ludwig-ai.github.io/ludwig-docs/latest/examples/titanic/).

# Import required libraries
import logging
import os
import shutil

import yaml

from ludwig.api import LudwigModel
from ludwig.datasets import titanic

# clean out prior results
shutil.rmtree("./results", ignore_errors=True)

# Download and prepare the dataset
training_set, test_set, _ = titanic.load(split=True)
print(training_set)


# Define Ludwig model object that drive model training
model = LudwigModel(config="./config.yml", logging_level=logging.INFO)

# initiate model training
(
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
    output_directory,  # location of training results stored on disk
) = model.train(
    dataset=training_set, experiment_name="simple_experiment", model_name="simple_model", skip_save_processed_input=True
)

# list contents of output directory
print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)

# batch prediction
model.predict(test_set, skip_save_predictions=False)
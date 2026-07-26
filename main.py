#
# Title : Deep Learning Benchmark for Crop Classification Using Sentinel-2 Time-Series Data
# 

import os
import argparse
import time
from pathlib import Path

# Local Functions
from pipeline import (
    Pipe_MakeCachData,
    Pipe_Training, 
    Pipe_Inference
)
from core.fileIF import read_yaml, write_yaml

from enum import IntEnum
class Step(IntEnum):
    S_DATASET = 0
    S_TRAINING = 1
    S_INFERENCE = 2


def ElapsedTime(diff_seconds):
    time_seconds = diff_seconds
 
    ### Day
    day = time_seconds // (24 * 3600)
    # Update the time variable to hold the remaining seconds after subtracting full days.
    time_seconds = time_seconds % (24 * 3600)
    ### Hour
    # Calculate the number of full hours in the remaining time.
    hour = time_seconds // 3600
    # Update the time variable to hold the remaining seconds after subtracting full hours.
    time_seconds %= 3600
    ### Minute
    # Calculate the number of full minutes in the remaining time.
    minutes = time_seconds // 60
    # Update the time variable to hold the remaining seconds after subtracting full minutes.
    time_seconds %= 60
    ### Second
    # The 'time' variable now represents the remaining seconds, which is the number of seconds.
    seconds = time_seconds
    
    # Print the time duration
    if day > 0:
        print('Elapsed time : {} day {} hr {} min {:.4f} sec \n'.format(int(day), int(hour), int(minutes), seconds))
    else:
        print('Elapsed time : {} hr {} min {:.4f} sec \n'.format(int(hour), int(minutes), seconds))


##############################################################################
# MAIN
##############################################################################
if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser(description='Main script for Crop Type Classification using TimeSen2Crop')
    parser.add_argument('--yaml_file',  type=str,   default='./main.yaml', help='yaml file path')
    args = parser.parse_args()
    print(args)

    config = read_yaml(args.yaml_file)

    prm_proc_name = config["Data"]["proc_name"]
    prm_proc_type = config["Data"]["proc_type"]
    prm_input_data = config["Data"]["dir_input_data"]
    prm_cache_data = config["Data"]["dir_cache_data"]
    prm_tile_valid = config["Data"]["tile_name_valid"]
    prm_tile_test = config["Data"]["tile_name_test"]
    prm_band_array = config["Data"]["band_array"]
    prm_add_doy = config["Data"]["enc_mode"]
    prm_dir_results = config["Data"]["dir_results"]

    prm_model_type = config["Model"]["model_type"]
    prm_hidden_dim = config["Model"]["hidden_dim"]
    prm_att_mode = config["Model"]["agg_mode"]
    prm_class_weights = config["Model"]["class_weights"]
    prm_batch_size = config["Model"]["batch_size"]
    prm_n_epoch = config["Model"]["n_epoch"]
    prm_es_patience = config["Model"]["es_patience"]
    prm_learning_rate = config["Model"]["learning_rate"]
    prm_weight_decay = config["Model"]["weight_decay"]

    save_dir = os.path.join(prm_dir_results, prm_proc_name)

    match prm_proc_type:
        case Step.S_DATASET:
            # [1] Making Cached Data (CSV -> PT)
            Pipe_MakeCachData(prm_input_data, prm_cache_data, prm_band_array,
                                prm_tile_valid, prm_tile_test)

        case Step.S_TRAINING:
            # [2] Training and Validation
            os.makedirs(save_dir, exist_ok=True)
            Pipe_Training(save_dir, prm_cache_data, 
                                prm_add_doy, prm_model_type, 
                                prm_hidden_dim, prm_att_mode, 
                                prm_class_weights, prm_batch_size,
                                prm_learning_rate, prm_weight_decay, 
                                prm_n_epoch, prm_es_patience)
            filepath = os.path.join(save_dir, args.yaml_file)
            write_yaml(filepath, config)

        case Step.S_INFERENCE:
            # [3] Inference using Test data
            Pipe_Inference(save_dir, prm_cache_data, prm_add_doy, 
                                prm_model_type, prm_hidden_dim, 
                                prm_att_mode, prm_batch_size)

    ElapsedTime(time.time()-start)


import os
from tqdm import tqdm

# Local Functions
from dataset import TimeSen2CropCacheBuilder
from core.utils import (
    ReadDataset, 
    MakeLabels,
    SplitDataset,
    SplitDataset_RandData
)

##############################################################################
# [1] Making Cached Data (CSV -> PT)
##############################################################################
def Pipe_MakeCachData(data_dir, cache_dir, band_array,
                    tile_name_valid, tile_name_test):
    print("<<< Making cached data started. >>>")

    # Making Dataset / Normalization
    # Read CSV file
    samples = ReadDataset(data_dir)
    # Make Labels
    class_to_idx, idx_to_class = MakeLabels(samples)
    # Split Dataset
    train_samples, valid_samples, test_samples = SplitDataset(
                                                    samples, 
                                                    tile_name_valid, 
                                                    tile_name_test)
    if (len(valid_samples) == 0) or (len(test_samples) == 0):
        # Random Selection
        train_samples, valid_samples, test_samples = SplitDataset_RandData(samples)

    print("<Train Data>")
    train_builder = TimeSen2CropCacheBuilder(
        samples=train_samples,
        class_to_idx=class_to_idx,
        out_dir=cache_dir,
        arr_bands=band_array,
        ds_name="train"
    )
    train_builder.build_cache_and_stats()

    print("<Valid Data>")
    valid_builder = TimeSen2CropCacheBuilder(
        samples=valid_samples,
        class_to_idx=class_to_idx,
        out_dir=cache_dir,
        arr_bands=band_array,
        ds_name="valid"
    )
    valid_builder.build_cache_and_stats()

    print("<Test Data>")
    test_builder = TimeSen2CropCacheBuilder(
        samples=test_samples,
        class_to_idx=class_to_idx,
        out_dir=cache_dir,
        arr_bands=band_array,
        ds_name="test"
    )
    test_builder.build_cache_and_stats()

    print("Making cached data has been finished.\n")

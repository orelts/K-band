# Convert raw h5 data into network data input, with support for cropping and customizable dataset sizes. We enable external cropping in the image domain in order to reduce the data size, as this enables fitting more unrolls on a GPU.
# The default cropped size is 400*300 for knee data and is 320*230 for brain data. These sizes enable the sample to keep most of the structures and only crop the edges, therefore not having too much impact on the training effect. But the cropped size can be re-defined in the configurations.
# @author: Frederic Wang, UC Berkeley, 2022

import os
from optparse import OptionParser

import h5py
import numpy as np
from utils_pipeline import fft2c, ifft2c


def create_input(input_dir, num_samples, config, data_type, is_test=False):
    """Generates input arrays of images, sensitivity maps, and kspace for the network
    Args:
        input_dir (string): path to processed data.
        num_samples (int): number of samples to create.
        config (dict): input generation config, reference line 96 for an example.
        data_type (string): knee or brain
        is_test (boolean): if generating inference data. If so, do not crop.
    Returns:
        imgs, sensitivity maps, ksp (ndarray)"""

    data_files = os.listdir(input_dir)
    data_files.sort()
    assert len(data_files) >= num_samples

    if is_test:
        # Getting dimensions for arrays from the file (not cropping if is_test)
        h5_data = h5py.File(os.path.join(input_dir, data_files[0]), "r")
        config["height"] = h5_data["kspace"].shape[1]
        config["width"] = h5_data["kspace"].shape[2]
        h5_data.close

    # Initialize all arrays we'll need. Images are [slices, X, Y], while sensitivity maps and k-space are [slices, coils, X, Y].
    imgs = np.zeros( (num_samples, config["height"], config["width"]), dtype=np.complex64 )
    maps = np.zeros( (num_samples, config["coils"], config["height"], config["width"]), dtype=np.complex64, )
    ksp = np.zeros( (num_samples, config["coils"], config["height"], config["width"]), dtype=np.complex64, )

    # Create vardens/k-band masks and convert data into deepinpy format.
    for i in range(num_samples):
        # Load images, k-space and sensitivity maps from fastMRI data.
        print("Preparing", data_files[i])
        h5_data = h5py.File(os.path.join(input_dir, data_files[i]), "r")
        if not is_test:
            n_coils = h5_data["kspace"].shape[0]
            orig_H = h5_data["kspace"].shape[1]
            orig_W = h5_data["kspace"].shape[2]
            mid_H = orig_H // 2
            mid_W = orig_W // 2

            if data_type == "brain":
                mid_W -= 5

            # Crop if height and width specified in config is less than the height and width of loaded data.
            imgs[i] = h5_data["target"][:][
                mid_H - (config["height"] // 2) : mid_H + (config["height"] // 2),
                mid_W - (config["width"] // 2) : mid_W + (config["width"] // 2),
            ]
            maps[i] = h5_data["sensmaps"][:][
                :,
                mid_H - (config["height"] // 2) : mid_H + (config["height"] // 2),
                mid_W - (config["width"] // 2) : mid_W + (config["width"] // 2),
            ]
            # Because the cropping is applied in the image domain, we need to re-compute the k-space of the cropped images
            for coil in range(n_coils):
                ksp[i][coil] = fft2c(
                    ifft2c(h5_data["kspace"][coil])[
                        mid_H
                        - (config["height"] // 2) : mid_H
                        + (config["height"] // 2),
                        mid_W - (config["width"] // 2) : mid_W + (config["width"] // 2),
                    ]
                )


            print(i + 1, "out of", num_samples, "samples done")
        else:
            imgs[i] = h5_data["target"]
            maps[i] = h5_data["sensmaps"]
            ksp[i] = h5_data["kspace"]

        h5_data.close()

    return imgs, maps, ksp


def get_args():
    parser = OptionParser()
    parser.add_option( "-c", "--config", dest="config", default="brain", 
                      help="Which fastMRI dataset to create data from, knee or brain or other.")
    parser.add_option("--train", dest="load_train", default="/home/orel/projects/K-band/data/brain_multicoil_train_processed", 
                      help="Folder directory contains processed(after generate_maps) train dataset (h5 format)")
    parser.add_option("--test", dest="load_test", default="/home/orel/projects/K-band/data/brain_multicoil_test_processed", 
                      help="Folder directory contains processed(after generate_maps) test dataset (h5 format)")
    parser.add_option("--target", dest="target", default="/home/orel/projects/K-band/data/brain_multicoil_target/", 
                      help="Folder directory contains target data (h5 format)")
    parser.add_option("--num_samples_train", dest="num_samples_train", default=2000, type=int, 
                      help="number of train samples to create. < num of files")
    parser.add_option("--num_samples_test", dest="num_samples_test", default=400, type=int,
                    help="number of test samples to create. < num of files")

    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    args = get_args()

    # The ratio of training samples in the whole dataset.
    # (1 - train_proportion) is the ratio of validation samples in the whole dataset.
    train_proportion = 0.8

    # Knee and Brain data configs from FastMRI
    fastMRI_brain_config = {
        "load_train": args.load_train,
        "load_test": args.load_test,
        "target": args.target,
        "num_samples_train": args.num_samples_train,
        "num_samples_test": args.num_samples_test,
        "height": 320,
        "width": 230,
        "coils": 16,
    }
    fastMRI_knee_config = {
        "load_train": args.load_train,
        "load_test": args.load_test,
        "target": args.target,
        "num_samples_train": args.num_samples_train,
        "num_samples_test": args.num_samples_test,
        "height": 400,
        "width": 300,
        "coils": 15,
    }
    # Empty template
    other_config = {
        "load_train": "",
        "load_test": "",
        "target": "",
        "num_samples_train": 0,
        "num_samples_test": 0,
        "height": 0,
        "width": 0,
        "coils": 0,
    }


    if args.config == "knee":
        config = fastMRI_knee_config
    elif args.config == "brain":
        config = fastMRI_brain_config
    elif args.config == "other":
        config = other_config
    else:
        raise ValueError(
            "Unknown config type specified. Must be 'knee' or 'brain' or 'other'"
        )
    train_data_file_name = (
        args.config
        + "_data_"
        + str(round(train_proportion * config["num_samples_train"]))
        + "samples_train.h5"
    )
    val_data_file_name = (
        args.config
        + "_data_"
        + str(round((1 - train_proportion) * config["num_samples_train"]))
        + "samples_val.h5"
    )
    test_data_file_name = (
        args.config
        + "_data_"
        + str(round(config["num_samples_test"]))
        + "samples_test.h5"
    )

    print("Generating training/validation data...")
    imgs, maps, ksp = create_input(
        config["load_train"], config["num_samples_train"], config, args.config
    )

    print("Generating test data...")
    test_imgs, test_maps, test_ksp = create_input(
        config["load_test"], config["num_samples_test"], config, args.config, True
    )

    # Split generated data into training and validation/test.
    train_ksp = ksp[: int(len(ksp) * train_proportion)]
    train_maps = maps[: int(len(maps) * train_proportion)]
    train_imgs = imgs[: int(len(imgs) * train_proportion)]

    val_ksp = ksp[int(len(ksp) * train_proportion):]
    val_maps = maps[int(len(maps) * train_proportion):]
    val_imgs = imgs[int(len(imgs) * train_proportion):]

    print("Saving files...")
    f = h5py.File(os.path.join(config["target"], train_data_file_name), "w")
    f.create_dataset("imgs", data=train_imgs, dtype=np.complex64)
    f.create_dataset("ksp", data=train_ksp, dtype=np.complex64)
    f.create_dataset("maps", data=train_maps, dtype=np.complex64)
    f.close()

    f = h5py.File(os.path.join(config["target"], val_data_file_name), "w")
    f.create_dataset("imgs", data=val_imgs, dtype=np.complex64)
    f.create_dataset("ksp", data=val_ksp, dtype=np.complex64)
    f.create_dataset("maps", data=val_maps, dtype=np.complex64)
    f.close()

    f = h5py.File(os.path.join(config["target"], test_data_file_name), "w")
    f.create_dataset("imgs", data=test_imgs, dtype=np.complex64)
    f.create_dataset("ksp", data=test_ksp, dtype=np.complex64)
    f.create_dataset("maps", data=test_maps, dtype=np.complex64)
    f.close()

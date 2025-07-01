# MONAI_101
Example repository with a MONAI 101 implementation. This is based on the [MONAI 101 tutorial notebook](https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/monai_101.ipynb).

## Training

To train the model and produce the model weight file, first install the requirements:

`pip install -r requirements.txt`

Next, run the training script, optionally setting the environment variable `MONAI_DATA_DIRECTORY` giving the
directory to download the MedNIST dataset to:

`python monai_101_train.py`

## Inference

The inference script can be run from its directory, and looks for all JPEG files in the directory given as the first
and only command line argument:

`python monai_101_infer.py .`

## Docker

Build the image with:

`docker build . -t monai_101`

This will run the inference script with `/inputs` as the directory to search for images in, so should be mounted with 
`-v` when running, such as the following which will do inference on the images in this repo:

`docker run --rm -v $(pwd):/inputs monai_101`

To use CUDA, GPUs must be made available within the running container:

`docker run --rm --gpus all -v $(pwd):/inputs monai_101`

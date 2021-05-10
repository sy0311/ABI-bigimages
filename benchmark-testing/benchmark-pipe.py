import numpy as np
import image3D
import logging
import time
import glob
import os
import tracemalloc

"""
This file was used to test the computational cost (time and memory) of:
- Reading TIFF files
- Saving image data into ZARR files
- Downsampling
- Saving image data into TIFF files
- Saving image data into ZARR files
- Reading ZARR files
"""

# ----------------------------------------------------------------------------------------------------------------------

# initating tests

downsample_no = 0

test_downsample = True
saveastif = False
saveaszarr = True
test_zarrfile = False

dir_main = 'benchmarking/6. Big/'
readtif_dir = [dir_main + 'file_0/out/38x94x94/', dir_main + 'file_0/out/75x188x188/', dir_main + 'file_0/out/150x375x375/',
        dir_main + 'file_0/out/300x750x750/', dir_main + 'file_0/out/450x1125x1125/', dir_main + 'file_0/out/600x1500x1500/']

readzarr_dir = 'zarr_out.zarr'

# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
#formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
formatter = logging.Formatter("%(asctime)s; %(levelname)s; %(message)s", "%m-%d-%Y %I:%M:%S %p")
file_handler = logging.FileHandler('%stest_out_append.log' % dir_main)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info('****************** Run benchmark_testing.py ******************')

# ----------------------------------------------------------------------------------------------------------------------


def read_tif_test(directory):
    """ Benchmark testing for reading TIFF files."""
    # start recording
    tracemalloc.start()
    start = time.process_time()

    # read tif files
    a = image3D.read_tif_dir_append(directory)

    # stop recording
    stop_time = time.process_time() - start
    current, peak = tracemalloc.get_traced_memory()

    # logging
    logger.info(
        '***** Read directory %s. Time taken (s): %s *****' % (directory, str(stop_time)))
    logger.info(
        '***** Current memory usage: %s MB; Peak: %s MB *****'
        % (str(current / 10 ** 6), str(peak / 10 ** 6)))

    return a


def save_zarr_test(a, filename, directory):
    """ Benchmark testing for saving image data into ZARR files."""
    # define filename
    # dir_out = '%sout2/file_%d/' % (dir_main, j)
    # fn_out = '%dx%dx%d' % (a.array.shape[0], a.array.shape[1], a.array.shape[2])
    # filename_out = fn_out + '.zarr'

    # start recording
    tracemalloc.start()
    start = time.process_time()

    # save file
    a.save(filename=filename, directory=directory)

    # stop recording
    stop_time = time.process_time() - start
    current, peak = tracemalloc.get_traced_memory()

    # logging
    logger.info('Saved layer to a zarr file. Time taken (s): %s' % (str(stop_time)))
    logger.info('Current memory usage: %s MB; Peak: %s MB' % (str(current / 10 ** 6), str(peak / 10 ** 6)))


def downsample_test(a):
    """ Benchmark testing for downsampling."""
    # start recording
    tracemalloc.start()
    start = time.process_time()

    # downsample
    a.downsample()

    # stop recording
    stop_time = time.process_time() - start
    current, peak = tracemalloc.get_traced_memory()

    # logging
    logger.info('Downsampling of layer %d, res %s complete. Time taken (s): %s'
                % (a.layer, str(a.array.shape), str(stop_time)))
    logger.info('Current memory usage: %s MB; Peak: %s MB' % (str(current / 10 ** 6), str(peak / 10 ** 6)))

    return a


def saveas_tif_test(filename, directory):
    """ Benchmark testing for saving image data into TIFF files."""
    #filename_out = fn_out + '.tif'
    start = time.process_time()
    a.saveas_tif(filename=filename, directory=directory)
    logger.info('Saved layer to tif. Time taken (s): %s' % (str(time.process_time() - start)))


def saveas_zarr_test(filename, directory):
    """ Benchmark testing for saving image data into ZARR files."""
    #filename_out = zarr_fn_out + '.zarr'
    start = time.process_time()
    a.save(filename=filename, directory=directory)
    logger.info('Saved layer to a zarr file. Time taken (s): %s' % (str(time.process_time() - start)))


def read_zarr_test(dir_in):
    """ Benchmark testing for reading ZARR files."""

    # get list of zarr files
    file_list = glob.glob(dir_in + "*.zarr")

    for j, filename in enumerate(file_list):

        # start recording
        tracemalloc.start()
        start = time.process_time()

        # load zarr file
        a = image3D.load(filename=filename)

        # stop recording
        stop_time = time.process_time() - start
        current, peak = tracemalloc.get_traced_memory()

        # logging
        logger.info('Reading zarr file %s. Created image3D object with array %s. Time taken (s): %s'
                    % (filename, str(a.array), str(stop_time)))
        logger.info(
            '***** Current memory usage: %s MB; Peak: %s MB *****'
            % (str(current / 10 ** 6), str(peak / 10 ** 6)))

        # path = dir_in + filename
        #
        # if os.path.isdir(path):
        #     for k, file in enumerate(os.listdir(path + '/')):
        #         if file.endswith(".zarr"):
        #             start = time.process_time()
        #             a = image3D.load(filename=file, directory=path + '/')
        #             logger.info('Reading zarr file %s/%s. Created image3D object with array %s. Time taken (s): %s'
        #                         % (filename, file, str(a.array), str(time.process_time() - start)))

# ----------------------------------------------------------------------------------------------------------------------

for j, dir_in in enumerate(readtif_dir):

    if test_downsample:

        # read TIFF file into an Array3D object
        a = read_tif_test(directory=dir_in)

        # save Array3D object into a ZARR file
        if saveaszarr:

            # define filenames
            # dir_out = '%sfile_%d/' % (dir_main, j)
            # zarr_fn_out = '%dx%dx%d.zarr' % (a.array.shape[0], a.array.shape[1], a.array.shape[2])

            # save to zarr file
            save_zarr_test(a, filename='zarr_out.zarr', directory='benchmarking/6. Big/file_0/out/')

        # perform downsampling steps
        for i in range(downsample_no):
            a = downsample_test(a)

            # define filenames
            dir_out = '%sfile_%d/' % (dir_main, j)
            tif_fn_out = '%dx%dx%d.tif' % (a.array.shape[0], a.array.shape[1], a.array.shape[2])
            zarr_fn_out = '%dx%dx%d.zarr' % (a.array.shape[0], a.array.shape[1], a.array.shape[2])

            # save downsampled array as a TIFF file
            if saveastif:
                saveas_tif_test(filename=tif_fn_out, directory=dir_out)

            # save downsampled array as a ZARR file
            if saveaszarr:
                saveas_zarr_test(filename=zarr_fn_out, directory=dir_out)

    # read ZARR files into an Array3D object
    if test_zarrfile:
        read_zarr_test(readzarr_dir)

import image3D
import logging
import time
import tracemalloc

"""
This file was used to benchmark test reading and writing image data using different chunk sizes
"""

# ----------------------------------------------------------------------------------------------------------------------
# FILE AND DIRECTORY SET UP

dir_main = 'benchmarking/6. Big/'
dir_in = [dir_main + 'file_0/out/38x94x94/', dir_main + 'file_0/out/75x188x188/',
          dir_main + 'file_0/out/150x375x375/', dir_main + 'file_0/out/300x750x750/',
          dir_main + 'file_0/out/450x1125x1125/', dir_main + 'file_0/out/600x1500x1500/',
          dir_main + 'file_0/out/900x2250x2250/']
# dir_in = [dir_main + 'file_0/out/600x1500x1500/', dir_main + 'file_0/out/900x2250x2250/']
file_out = 'benchmarking/6. Big/out/18_feb/zarr_out.zarr'
log_file = 'benchmarking/6. Big/out/18_feb/benchmark_chunks_03.log'

chunksizes = [(20, 20, 20), (50, 50, 50), (100, 100, 100), (200, 200, 200), (500, 500, 500),
              (800, 800, 800), (1000, 1000, 1000), (1200, 1200, 1200), (1500, 1500, 1500)]
# chunksizes = [(1500, 1500, 1500), (2250, 2250, 2250)]
trial_no = 3

# ----------------------------------------------------------------------------------------------------------------------
# LOGGING SETUP

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s; %(levelname)s; %(message)s", "%m-%d-%Y %I:%M:%S %p")
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS FOR TIMING

def dask_readwrite(dir_in, file_out, **kwargs):

    # start recording
    tracemalloc.start()
    start = time.process_time()

    # read tif files
    a = image3D.tif_to_zarr(dir_in=dir_in, file_out=file_out, **kwargs)

    # stop recording
    stop_time = time.process_time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # logging
    logger.info('Read and Write time (s): %s' % (str(stop_time)))
    logger.info('Peak memory usage (MB): %s' % (str(peak / 10 ** 6)))

# ----------------------------------------------------------------------------------------------------------------------
# BENCHMARKING CODE


for i in range(trial_no):
    logger.info('****************************************************************')
    logger.info('******************** BENCHMARK TRIAL NO. %02d ********************' % (i + 1))
    logger.info('****************************************************************')

    for dir in dir_in:

        logger.info('*************** FILE IN: %s ***************' % dir)

        for chunks in chunksizes:
            try:
                logger.info('****** CHUNK SIZE: %s ******' % str(chunks))
                dask_readwrite(dir_in=dir, file_out=file_out, chunks=chunks)
            except:
                logger.info('ERROR with chunk size: %s' % str(chunks))

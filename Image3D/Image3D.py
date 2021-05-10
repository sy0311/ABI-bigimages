import numpy as np
import math
from statistics import mean
import zarr
import tifffile as tiff
import os
import glob
import imageio
import dask
import dask.array as da

class Array3D(object):
    """Object that stores Zarr array data as an image pyramid.

    Parameters
    ----------
    zarr_array : ZarrArray object
        Array store
    scale_layer : int, optional
        Scale layer number, zero (default) at the original resolution and increasing with downsampling.

    Methods
    -------
    create_pyramid
    downsample
    save
    saveas_tif
    saveas_tif_dir

    """

    def __init__(self, zarr_array, scale_layer=0):

        # define attributes
        self.array = zarr_array     # Zarr array of image data
        self.layer = scale_layer    # 0 as the original layer and increased by one with downsampling

    def create_pyramid(self, layers=1, directory='out/'):
        """ Performs several downsampling steps to create an image pyramid """
        for i in range(layers):
            self.downsample()
            filename = 'RES(%dx%dx%d)' % (self.array.shape[0], self.array.shape[1], self.array.shape[2])
            self.save(filename=filename, directory=directory)

    def downsample(self):
        """ Performs one downsampling step on the Array3D object."""
        Z = self.array.shape[0]
        Y = self.array.shape[1]
        X = self.array.shape[2]

        for z in range(-(-Z // 2)):
            for x in range(X // 2):
                # perform downsampling on each 4xYx4 voxel section
                self.array[z, :-(-Y // 2), x] = downsample3d(np.array(self.array[z*2:z*2+2, :, x*2:x*2+2]))
            if X % 2 != 0:
                # downsampling when there's an odd number of voxels in the X direction
                self.array[z, :-(-Y // 2), X // 2] = downsample2d(np.array(self.array[z*2:z*2+2, :, -1]))

        # reduce the array size by half (+1 if odd)
        self.array.resize(-(-np.array(self.array.shape) // 2))

        # increment layer attribute
        self.layer += 1

    def save(self, filename, directory=''):
        """Convenience function to save the array attribute to the local file system as a nested zarr file.

        Parameters
        ----------
        filename : string
            Name of zarr file.
        directory : string
            Path to directory in file system
        """

        zarr.save(zarr.NestedDirectoryStore('%s%s' % (directory, filename)), self.array)

    def saveas_tif(self, filename, directory=''):
        """Convenience function to save the array attribute to the local file system as a single tiff file.

        Parameters
        ----------
        filename : string
            Name of tiff file.
        directory : string
            Path to directory in file system

        """

        directory = directory + 'out/'

        if not os.path.isdir(directory):
            os.makedirs(directory)

        tiff.imsave('%s%s' % (directory, filename), np.array(self.array))

    def saveas_tif_dir(self, filename, directory=''):
        """Convenience function to save the array attribute to the local file system as an image sequence of 2D tiff
        files.

        Parameters
        ----------
        filename : string
            Name of tiff file.
        directory : string
            Path to directory in file system

        """

        directory = "%s%dx%dx%d/" % (directory, self.array.shape[0], self.array.shape[1], self.array.shape[2])

        if not os.path.isdir(directory):
            os.makedirs(directory)

        for i in range(self.array.shape[0]):
            a = ("%s%s_%04d.tif" % (directory, filename, i))
            tiff.imsave(a, np.array(self.array[i,:,:]))


def read_3d_tif(filename, directory=None, **kwargs):
    """Convenience function to create an Array3D object from a single 3D tiff file.

    Parameters
    ----------
    filename : string
        Path to directory in file system of tif file.
    **kwargs
        Additional parameters are passed through to :func:`zarr.create`

    Returns
    -------
    a: :class:`image3D.Array3D`

    """

    return create(tiff.imread('%s%s' % (directory, filename)), **kwargs)


def read_tif_dir_numpy(directory, **kwargs):
    """ Reads tiff files from a directory and returns a Array3D object of the image data through the <NumPy method>.

    This function is used for the situation where there are multiple 2D images in one folder, with each folder
    representing a layer of the 3D image.
    """
    # list that stores image data
    a = []

    # append data from each TIFF file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            a.append(tiff.imread(directory + filename))

    # convert into a NumPy array and return the Array3D object
    return create(np.array(a), **kwargs)


def read_tif_dir_indexing(directory, **kwargs):
    """ Reads tiff files from a directory and returns a Array3D object of the image data through the
    <Zarr-indexing method>.

    This function is used for the situation where there are multiple 2D images in one folder, with each folder
    representing a layer of the 3D image.
    """
    # list of TIFF filenames from directory
    file_list = glob.glob(directory + "*.tif")

    # initialise Array3D object
    a = directory + file_list[0]
    sample = tiff.imread(file_list[0])
    shape = (len(file_list), sample.shape[0], sample.shape[1])
    z = create(zarr.zeros(shape), **kwargs)
    z.array[0, :, :] = sample
    file_list.pop(0)

    # read and store image data via indexing
    for i, filename in enumerate(file_list):
        z.array[i+1,:,:] = tiff.imread(filename)

    return z

def read_tif_dir_append(directory, **kwargs):
    """ Reads tiff files from a directory and returns a Array3D object of the image data through the
    <Zarr-appending method>.

    This function is used for the situation where there are multiple 2D images in one folder, with each folder
    representing a layer of the 3D image.
    """
    # list of TIFF filenames from directory
    file_list = glob.glob(directory + "*.tif")

    # initialise Array3D object
    sample = tiff.imread(file_list[0])
    sample = np.reshape(sample, (1,) + sample.shape)
    sample_shape = sample.shape
    z = create(sample, **kwargs)
    file_list.pop(0)

    # read and store image data via appending
    for filename in file_list: # is this in order?
        z_temp = tiff.imread(filename)
        z_temp = np.reshape(z_temp, sample_shape)
        z.array.append(z_temp, axis=0)

    return z


def tif_to_zarr(dir_in, file_out, dir_out='', chunks=(64, 64, 64), **kwargs):
    """ Reads tiff files from a directory and returns a Array3D object of the image data through the
    <Dask method>.
    """
    # list of TIFF filenames from directory
    file_list = glob.glob(dir_in + "*.tif")

    # open image data in a dask array
    sample = imageio.imread(file_list[0])
    lazy_list = [dask.delayed(imageio.imread)(fn) for fn in file_list]
    lazy_array = [da.from_delayed(image, shape=sample.shape, dtype=sample.dtype) for image in lazy_list]
    x = da.stack(lazy_array[:])
    x = x.rechunk(chunks=chunks)

    # save dask array to zarr file
    x.to_zarr(url=zarr.NestedDirectoryStore(dir_out + file_out), overwrite=True, **kwargs)

    return load(filename=file_out, directory=dir_out)


def create(arrayin, chunks=(64, 64, 64), store=None, filename=None, **kwargs):
    """ creates a Array3D object from an NumPy array """

    # initialise zarr array
    if store:
        store = zarr.NestedDirectoryStore(filename)

    return Array3D(zarr.array(arrayin, chunks=chunks, store=store, **kwargs), **kwargs)


def load(filename, directory=None, nested=True, **kwargs):
    """ loads a Zarr file into an Array3D object """

    if nested:
        store = zarr.NestedDirectoryStore('%s%s' % (directory, filename))
        return create(zarr.load(store), **kwargs)
    else:
        return create(zarr.load('%s%s' % (directory, filename)), **kwargs)


def saveas_csv(arrayin, directory=None):
    """Save NumPy array into a CSV file"""

    directory = directory + 'out/'

    if not os.path.isdir(directory):
        os.makedirs(directory)

    filename = ("%dx%dx%d" % (arrayin.shape[0], arrayin.shape[1], arrayin.shape[2]))

    for i in range(arrayin.shape[0]):
        np.savetxt("%s%s/%s_%04d.csv" % (directory, filename, filename, i), arrayin, delimiter=",")


def downsample3d(arrayin):
    """ returns a 2x downsampled array on an input array """

    arrayout = np.array([])

    for i in range(math.ceil(arrayin.shape[1] / 2)):
        arrayout = np.append(arrayout, calc_mean(arrayin[:, i * 2:i * 2 + 2,:]))

    return arrayout.astype(np.uint16)


def downsample2d(arrayin):
    """ returns a 2x downsampled array on an input array """

    arrayout = np.array([])

    for i in range(math.ceil(arrayin.shape[1] / 2)):
        arrayout = np.append(arrayout, calc_mean(arrayin[:, i * 2:i * 2 + 2]))

    return arrayout.astype(np.uint16)


def calc_mean(mean_array):
    """ returns the mean of an input array """

    return round(mean(mean_array.astype(float).flatten()))

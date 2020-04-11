# Only used in feature extraction


# import the necessary packages
import h5py
import os


class HDF5DatasetWriter:
    '''
    CONSTRUCTOR
    dimensions : dimension or shape of the data we will be storing in the dataset. Think of dimensions as the .shape of a NumPy array. 
        If we were storing the (ﬂattened) raw pixel intensities of the 28 × 28 = 784 MNIST dataset, then dimensions=(70000, 784) as there are 70,000 examples in MNIST, each with a dimensionality of 784.
        If we wanted to store the raw CIFAR-10 images, then we would set dimensions=(60000, 32, 32, 3) as there are 60,000 total images in the CIFAR-10 dataset, each represented by a 32×32×3 RGB image.

    output_path : path to where output HDF5 file will be stored to disk.

    data_key: (optional) name of the dataset that will store the data our algorithm will learn from

    buffer_size : (optional) size of our in-memory buffer, which we default to 1,000 feature vectors/images. Once we reach buffer_size, we’ll ﬂush the buffer to the HDF5 dataset.     
    '''
    def __init__(self, dimensions, output_path, data_key='images', buffer_size=1000):
        # check to see if the output path exists, and if so, raise an exception
        if os.path.exists(output_path):
            raise ValueError(
                "The supplied 'output path' already exists and cannot be overwritten. Manually delete the file before contuinuing", output_path)

        # open the HDF5 database for writing and create two datasets: one to store the images/feature and another to store class labels
        self.db = h5py.File(output_path, 'w')
        self.data = self.db.create_dataset(data_key, dimensions, dtype="float")
        self.labels = self.db.create_dataset("labels", (dimensions[0],), dtype="int")

        # store the buffer size, then initizlize the buffer itself along with the index into the database
        self.buffer_size = buffer_size
        self.buffer = {'data': [], 'labels': []}
        self.idx = 0


    """
    rows : rows to add to database

    labels : corresponding class labels for the rows
    """
    def add(self, rows, labels):
        # add the rows and labels to the buffer
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)

        # if buffer exceeds set limits flush it
        if len(self.buffer['data']) >= self.buffer_size:
            self.flush()

    """ write the buffers to disk then reset the buffer 
    """
    def flush(self):
        # determines the next available row in the matrix.
        i = self.idx + len(self.buffer['data'])
        # Write data to database
        self.data[self.idx:i] = self.buffer['data']
        self.labels[self.idx:i] = self.buffer['labels']
        # store next available position in database in idx variable
        self.idx = i
        # resets the buffers.
        self.buffer = {'data': [], 'labels': []}

    """ 
    Store the raw string names of the class labels in a separate dataset
    """
    def storeClassLabels(self, classLabels):
        # create a dataset to store the actual class label names, then store the class labels
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset(
            'label_names', (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    """
    write any data left in the buffers to HDF5 and as close the dataset:
    """
    def close(self):
        # check to see if ther is any other entries in the buffer that need to be 
        # flushed to the disk
        if len(self.buffer['data']) > 0:
            self.flush()

        # close the dataset
        self.db.close()

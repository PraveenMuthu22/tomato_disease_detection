# import the necessary packages
import h5py
import os


class HDF5DatasetWriter:
    '''
    CONSTRUCTOR
    dims : dimension or shape of the data we will be storing in the dataset. Think of dims as the .shape of a NumPy array. 
        If we were storing the (ﬂattened) raw pixel intensities of the 28 × 28 = 784 MNIST dataset, then dims=(70000, 784) as there are 70,000 examples in MNIST, each with a dimensionality of 784.
        If we wanted to store the raw CIFAR-10 images, then we would set dims=(60000, 32, 32, 3) as there are 60,000 total images in the CIFAR-10 dataset, each represented by a 32×32×3 RGB image.

    outputPath : path to where output HDF5 file will be stored to disk.

    dataKey: (optional) name of the dataset that will store the data our algorithm will learn from

    bufSize : (optional) size of our in-memory buffer, which we default to 1,000 feature vectors/images. Once we reach bufSize, we’ll ﬂush the buffer to the HDF5 dataset.     
    '''
    def __init__(self, dims, outputPath, dataKey='images', bufSize=1000):
        # check to see if the output path exists, and if so, raise an exception
        if os.path.exists(outputPath):
            raise ValueError(
                "The supplied 'output path' already exists and cannot be overwritten. Manually delete the file before contuinuing", outputPath)

        # open the HDF5 database for writing and create two datasets: one to store the images/feature and another to store class labels
        self.db = h5py.File(outputPath, 'w')
        self.data = self.db.create_dataset('labels', (dims[0],), int)

        # store the buffer size, then initizlize the buffer itself along with the index into the database
        self.bufSize = bufSize
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

        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer['data']) >= self.bufSize:
            self.flush()

    """ write the buffers to disk then reset the buffer 
    
    If we think of our HDF5 dataset as a big NumPy array, then we need to keep track of the 
    current index into the next available row where we can store data (without overwriting 
    existing data) –
    """
    def flush(self):
        # determines the next available row in the matrix.
        i = self.idx + len(self.buffer['data'])
        # apply NumPy array slicing to store the data and labels in the buffers.
        self.data[self.idx:i] = self.buffer['data']
        self.labels[self.idx:i] = self.buffer['labels']
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
    Finally, our last function close will be used to write any data left in the buffers to
     HDF5 as well as close the dataset:
    """
    def close(self):
        # check to see if ther is any other entries in the buffer that need to be flushed to the disk
        if len(self.buffer['data']) > 0:
            self.flush()

        # close the dataset
        self.db.close()

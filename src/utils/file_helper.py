import bz2
import _pickle as cPickle


def compress_pickle(title, data):
    # Pickle a file and then compress it into a file with extension 
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)

def decompress_pickle(file):
    # Load any compressed pickle file
    data = bz2.BZ2File(file + '.pbz2', 'rb')
    data = cPickle.load(data)
    return data        
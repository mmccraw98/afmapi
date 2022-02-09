from numpy import nan_to_num, array, sqrt
import pickle
import os
import re
from pandas import read_csv, read_excel, DataFrame
import os.path
from os.path import isfile
from afmapi.utils import selectyesno
from igor.binarywave import load as loadibw
from re import sub, search


def getnextfile(filename):
    '''
    given a tc_data name, with an extension, find the next numeric instance of the tc_data name
    i.e. the_file1.csv -> the_file2.csv
    :param file_name: str tc_data name with a tc_data extension
    :return: str the next numeric instance of the tc_data name
    '''
    name, extension = filename.split(sep='.')  # split the tc_data name at the tc_data extension
    # \d indicates numeric digits, $ indicates the end of the string
    stripped_name = re.sub(r'\d+$', '', name)  # remove any numbers at the end of the tc_data
    fnum = re.search(r'\d+$', name)  # get any numbers at the end of the tc_data
    # if there are any numbers at the end of the tc_data, add 1 to get the next tc_data number and cast it as a string
    # if there aren't any numbers at the end of the tc_data, use 1 as the next number
    next_fnum = str(int(fnum.group()) + 1) if fnum is not None else '1'
    return stripped_name + next_fnum + '.' + extension  # return the next tc_data number


def safesave(thing, path, overwrite=False):
    '''
    safely saves a thing to a given path, avoids overwrite issues
    :param thing: obj thing to be saved
    :param path: os.path path where the thing will be saved
    :param overwrite: bool determines whether or not to overwrite
    :return: none
    '''

    # defining some variables that will be used often
    dir_name = os.path.dirname(path)  # directory name
    file_name = os.path.basename(path)  # tc_data name

    # check if path exists and make it if it doesn't
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    # if path exists and the tc_data exists get the next available tc_data name and adjust the path to reflect the change of name
    # if overwrite is enabled, then skip the renaming step and just overwrite using the given path
    while os.path.isfile(path := os.path.join(dir_name, file_name)) and not overwrite:
        file_name = getnextfile(file_name)

    # get the tc_data extension and save the tc_data accordingly
    extension = file_name.split(sep='.')[-1]
    if extension == 'csv':
        thing.to_csv(path, index=False)
    elif extension == 'xlsx':
        thing.to_excel(path, index=False)
    elif extension == 'txt':
        with open(path, 'w') as f:
            f.write(thing)
    else:
        with open(path, 'wb') as f:
            pickle.dump(thing, f)


def get_files(dir, req_ext=None):
    '''
    gets all the files in the given directory
    :param dir: str directory from which you want to load files from
    :param req_ext: optional str required tc_data extension
    :return: list of str names of the files in the given directory
    '''
    if req_ext is None:
        return [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    else:
        return [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and req_ext in f]


def get_folders(dir):
    '''
    gets all the folders in the given directory
    :param dir: str directory from which you want the sub-directories
    :return: list of str names of the sub-directories
    '''
    return [f.path for f in os.scandir(dir) if f.is_dir()]


def load(path, required_extension=None):
    '''
    loads data from a number of formats into python
    :param path: str path to thing being loaded in
    :param required_extension: str required extension for the file(s) to be loaded
    i.e. only load files with the required_extension
    :return: the data
    '''
    if not os.path.isfile(path):
        exit('data does not exist')
    file_name = os.path.basename(path)  # tc_data name
    extension = file_name.split(sep='.')[-1]
    if extension == 'csv' or required_extension == 'csv':
        data = read_csv(path)
    elif extension == 'xlsx' or required_extension == 'xlsx':
        data = read_excel(path, engine='openpyxl')
    elif extension == 'txt' or required_extension == 'txt':
        with open(path, 'r') as f:
            data = f.read()
    elif extension == 'pkl' or required_extension == 'pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        exit('extension not yet supported: {}'.format(file_name))
    return data


# below has been stripped from https://github.com/N-Parsons/ibw-extractor
# is used in bulk_ibw_extractor.py to extract ibw files into csv format

def from_repr(s):
    """Get an int or float from its representation as a string"""
    # Strip any outside whitespace
    s = s.strip()
    # "NaN" and "inf" can be converted to floats, but we don't want this
    # because it breaks in Mathematica!
    if s[1:].isalpha():  # [1:] removes any sign
        rep = s
    else:
        try:
            rep = int(s)
        except ValueError:
            try:
                rep = float(s)
            except ValueError:
                rep = s
    return rep


def fill_blanks(lst):
    """Convert a list (or tuple) to a 2 element tuple"""
    try:
        return (lst[0], from_repr(lst[1]))
    except IndexError:
        return (lst[0], "")


def flatten(lst):
    """Completely flatten an arbitrarily-deep list"""
    return list(_flatten(lst))


def _flatten(lst):
    """Generator for flattening arbitrarily-deep lists"""
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        elif item not in (None, "", b''):
            yield item


def process_notes(notes):
    """Splits a byte string into an dict"""
    # Decode to UTF-8, split at carriage-return, and strip whitespace
    note_list = list(map(str.strip, notes.decode(errors='ignore').split("\r")))
    note_dict = dict(map(fill_blanks, [p.split(":") for p in note_list]))

    # Remove the empty string key if it exists
    try:
        del note_dict[""]
    except KeyError:
        pass
    return note_dict


def ibw2dict(filename):
    """Extract the contents of an *ibw to a dict"""
    data = loadibw(filename)
    wave = data['wave']

    # Get the labels and tidy them up into a list
    labels = list(map(bytes.decode,
                      flatten(wave['labels'])))

    # Get the notes and process them into a dict
    notes = process_notes(wave['note'])

    # Get the data numpy array and convert to a simple list
    wData = nan_to_num(wave['wData']).tolist()

    # Get the filename from the file - warn if it differs
    fname = wave['wave_header']['bname'].decode()
    input_fname = os.path.splitext(os.path.basename(filename))[0]
    if input_fname != fname:
        print("Warning: stored filename differs from input file name")
        print("Input filename: {}".format(input_fname))
        print("Stored filename: {}".format(str(fname) + " (.ibw)"))

    return {"filename": fname, "labels": labels, "notes": notes, "data": wData}


def ibw2xlsx(ibw_file_name, overwrite=False):
    data = ibw2dict(ibw_file_name)
    # data['notes'] -> wavenotes
    headers = data['labels']
    outname = ibw_file_name.replace('ibw', 'xlsx')
    if isfile(outname):
        if overwrite:
            print('overwriting {}'.format(data['filename']))
        elif not selectyesno('{} exists. overwrite?'.format(data['filename'])):
            print('skipping')
            return 0
    DataFrame(data['data'], columns=headers).to_excel(outname, index=False)


def search_between(start, end, string):
    return search('%s(.*)%s' % (start, end), string).group(1)


def get_exp_notes(experiment_path):
    notes = [load(f) for f in get_files(experiment_path, 'txt') if 'notes' in f][0]

    r_unformatted = search_between('spherical radius: ', '\n', notes)
    r_scale = array([1e-12, 1e-9, 1e-6, 1e-3])[[n in sub("[0-9,.]+", " ", r_unformatted).lower()
                                                   for n in ['pm', 'nm', 'um', 'mm']]]
    r = float(sub("[a-z]+", " ", r_unformatted)) * r_scale[0]

    freq = float(sub("[a-z]+", " ", search_between('freq: ', 'Hz', notes)))
    if 'k' in sub("[0-9,.]+", " ", search_between('freq', 'Hz', notes)).lower():
        freq *= 1e3

    k = float(search_between('k: ', r'nN/nm', notes))

    return {'r': r, 'freq': freq, 'k': k, 'a': 4 * sqrt(r) / (3 * (1 - 0.5 ** 2))}

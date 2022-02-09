from afmapi.io import ibw2dict
from os import walk
from os.path import join, split
from pandas import DataFrame

req_labels = ['Raw', 'Defl', 'ZSnsr']

data_path = r''  # all the files are in some directory structure here
target_path = r''  # all the files will be saved in a single directory here

def file_extension(path):
    return split(path)[-1].split('.')[-1]

files = []
for (dirpath, dirnames, filenames) in walk(data_path):
    files += [join(dirpath, file) for file in filenames]
ibws = [f for f in files if file_extension(f) == 'ibw']
del files
if len(ibws) == 0:
    exit('path [{}] contains no ibw files or is invalid'.format(data_path))

f_count = 1
for i, f in enumerate(ibws):
    print('{:.1f}%'.format(i / len(ibws) * 100), end='\r')
    try:
        data = ibw2dict(f)
        headers = data['labels']
        if headers != req_labels:
            continue  # skips maps
        outname = join(target_path, f_count + '.xlsx')
        DataFrame(data['data'], columns=headers).to_excel(outname, index=False)
    except:
        pass  # because i am a horrible programmer

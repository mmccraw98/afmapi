from os import walk, remove
from os.path import join, split
import argparse
from afmapi.io import ibw2xlsx

# parser = argparse.ArgumentParser()
# parser.add_argument('path')
# args = parser.parse_args()
# path = args.path

path = r'C:\Users\Windows\PycharmProjects\mat-phys\data\gelma_master\afm'

def file_extension(path):
    return split(path)[-1].split('.')[-1]


files = []
for (dirpath, dirnames, filenames) in walk(path):
    files += [join(dirpath, file) for file in filenames]
ibws = [f for f in files if file_extension(f) == 'ibw']
del files
if len(ibws) == 0:
    exit('path [{}] contains no ibw files or is invalid'.format(path))

for i, f in enumerate(ibws):
    print('{:.1f}%'.format(i / len(ibws) * 100), end='\r')
    try:
        ibw2xlsx(f, overwrite=True)
        remove(f)
    except:
        print('issue with [{}]'.format(f))
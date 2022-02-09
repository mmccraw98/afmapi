from general import get_files
import os
from numpy import loadtxt
from pandas import DataFrame
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('folder')
# args = parser.parse_args()
# folder = args.folder

folder = r'C:\Users\Windows\PycharmProjects\mat-phys\data\220209_keidar_cells\sample_1\treated\ForceMap02'

try:
    files = get_files(folder, req_ext='txt')
except:
    exit('path [{}] contains no txt files or is invalid'.format(folder))

def get_matching_names(str_to_match, names):
    return [name for name in names if str_to_match in name]
def sort_data(curve_data, names):
    for curve, name in zip(curve_data, names):
        if 'Defl' in name:
            defl = curve
        elif 'ZSnsr' in name:
            zsnsr = curve
    return defl, zsnsr


strip_text = 'DeflRawZSnsr.txt'  # text to remove from the filenames
paths, file_names = zip(*[os.path.split(f) for f in files])  # returns two lists, one with the paths, one with the filenames
unique_file_names = [f for f in set([f.strip(strip_text) for f in file_names])]  # gets the unique filenames by stripping the useless text

for i, unique_name in enumerate(unique_file_names):
    print('Percent Complete: {:.0f}%'.format(100 * i / len(unique_file_names)), end='\r')
    try:
        curve_data, names = zip(*[(loadtxt(os.path.join(folder, name)), name) for name in get_matching_names(unique_name, file_names)])
        defl, zsnsr = sort_data(curve_data, names)
        DataFrame({'defl': defl, 'zsnsr': zsnsr}).to_excel(os.path.join(folder, unique_name + '.xlsx'), index=False)
    except:  # excel has some size limit which causes an issue here
        pass

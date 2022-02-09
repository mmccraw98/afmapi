from time import time
import psutil
import os
import os.path


def selectyesno(prompt):
    '''
    given a prompt with a yes / no input answer, return the boolean value of the given answer
    :param prompt: str a prompy with a yes / no answer
    :return: bool truth value of the given answer: yes -> True, no -> False
    '''
    print(prompt)  # print the user defined yes / no question prompt
    # list of understood yes inputs, and a list of understood no inputs
    yes_choices, no_choices = ['yes', 'ye', 'ya', 'y', 'yay'], ['no', 'na', 'n', 'nay']
    # use assignment expression to ask for inputs until an understood input is given
    while (choice := input('enter: (y / n) ').lower()) not in yes_choices + no_choices:
        print('input not understood: {} '.format(choice))
    # if the understood input is a no, it returns false, if it is a yes, it returns true
    return choice in yes_choices


def tic():
    global current_time
    current_time = time()


def toc(return_numeric=False):
    if return_numeric:
        return time() - current_time
    else:
        print('process completed in {:.2f}s'.format(time() - current_time))


def getmemuse(return_numeric=False):
    if return_numeric:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    else:
        print('{} mb used'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

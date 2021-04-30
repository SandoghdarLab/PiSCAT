import shutil
from pathlib import Path
from os.path import exists
import os
import subprocess
import sys


def parse_args():

    if len(sys.argv) <= 1:
        if confirm('No location given. Shall the tutorials be copied to '
                   'the current working directory? [y/n] '):
            target_root = Path.cwd()
        else:
            raise Exception('Exiting.')
    elif len(sys.argv) > 2:
        raise Exception('ERROR! Too many arguments!\n\n'
                        'USAGE: python -m piscat.Tutorials TARGET_DIRECTORY')
    else:
        target_root = Path(sys.argv[1])

    return target_root


def find_jupyter():
    jupyter = shutil.which('jupyter')
    if jupyter:
        jupyter += ' notebook'

    if not jupyter:
        jupyter = shutil.which('jupyter-notebook')

    if not jupyter:
        raise Exception('Jupyter not found on your system.')

    return jupyter


def start_server(root, jupyter):
    os.chdir(root)
    subprocess.run(jupyter, shell=True)


def confirm(prompt):
    answer = ''
    while answer.lower() != 'y' and answer.lower() != 'n':
        answer = input(prompt)
    return answer == 'y'


def copy_tutorials(target_root):

    target = target_root / 'Tutorials/JupyterFiles'
    src = Path(__file__).parent / 'JupyterFiles'

    if exists(target):
        Warning('Warning! Target directory %s already exists' % str(target))
    else:
        shutil.copytree(src, target)

    return target


def start_tutorials():

    try:
        target_root = parse_args()
        jupyter = find_jupyter()
        root = copy_tutorials(target_root)
        start_server(root, jupyter)
        print('1')
    except Exception as e:
        print(e)


start_tutorials()
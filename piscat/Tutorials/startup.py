import shutil
from pathlib import Path
from os.path import exists
import os
import subprocess
import sys


def parse_args():
    """Parses the command line arguments.

    The first command line argument is the directory where to copy
    the tutorials.

    If no command line argument is given, assume working directory
    as target, but ask for user confirmation.

    Raises
    ------
    Exception
        if command line arguments are invalid (e.g. more than one
        argument)

    """

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
    """Try to find the Jupyter executable on the system.

    Returns
    -------
    jupyter: str
        The command to start jupyter

    Raises
    ------
    Exception
        if no jupyter executable is found on the system.
    """

    jupyter = shutil.which('jupyter')
    if jupyter:
        jupyter += ' notebook'

    if not jupyter:
        jupyter = shutil.which('jupyter-notebook')

    if not jupyter:
        raise Exception('Jupyter not found on your system. Please start it manually.')

    return jupyter


def start_server(root, jupyter):
    """Start up Jupyter server.

    Parameters
    ----------
    root: pathlib.Path
        The directory in which to start the server.

    jupyter: str
        The console command to start the jupyter server.

    """
    os.chdir(root)
    subprocess.run('jupyter nbextension enable --py widgetsnbextension', shell=True)
    subprocess.run(jupyter, shell=True)


def confirm(prompt):
    """Get user confirmation from standard input.

    Parameters
    ----------
    prompt: str
        The message to show.

    Returns
    -------
    True or false depending on whether the user has answered with yes or no.
    """
    answer = ''
    while answer.lower() != 'y' and answer.lower() != 'n':
        answer = input(prompt)
    return answer == 'y'


def copy_tutorials(target_root):
    """Copies the PiSCAT tutorials to a given directory.

    Parameters
    ----------
    target_root: pathlib.Path
        Where to copy the directories. If already existing, files
        are not copied to prevent accidental overwriting.

    Returns
    -------
    target: pathlib.Path
        The path to the directory with the tutorials.
    """

    target = target_root / 'Tutorials/JupyterFiles'
    src = Path(__file__).parent / 'JupyterFiles'

    if exists(target):
        Warning('Warning! Target directory %s already exists.' % str(target))
    else:
        print('Copying tutorials to ' + str(target))
        shutil.copytree(src, target)

    return target


def start_tutorials():
    """Copy the PiSCAT tutorials to the target directory and start Jupyter server

    The target directory is the first argument of sys.argv.
    """

    target_root = parse_args()
    root = copy_tutorials(target_root)
    jupyter = find_jupyter()
    start_server(root, jupyter)

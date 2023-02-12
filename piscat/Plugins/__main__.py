from git import Repo
from pathlib import Path
from os.path import exists
import os
import sys
import subprocess

from piscat.Visualization.print_colors import PrintColors


def parse_args():
    """Parses the command line arguments.

    The first command line argument is the module name.

    Raises
    ------
    Exception
        if command line arguments are invalid (e.g. more than one
        argument)

    """

    if len(sys.argv) == 2:
        name = sys.argv[1]
    else:
        raise Exception('ERROR! Please specify plugin name')
    return name


class Plugin(PrintColors):
    def __init__(self):
        self.dic_git_url = {'UAI': 'https://github.com/SandoghdarLab/PiSCAT_plugin.git'}

    def confirm(self, prompt):
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

    def copy_plugin(self, plu_name):
        """Copies the PiSCAT plugin to a plugin directory.

        Parameters
        ----------
        plu_name: name of the plugin

        Returns
        -------
        target: boolean
            Specify and plugin data changed or not
        """
        target = os.path.dirname(os.path.realpath(__file__))
        target = os.path.join(target, plu_name)
        if exists(target):
            print('Warning! Target directory %s already exists.' % str(target))
            answer = self.confirm(f'{self.BOLD}\n Target directory %s already exists. '
                                  f'Do you want to pull the latest change (y/n): {self.ENDC}' % str(target))
            if answer:
                try:
                    repo = Repo(target)
                    o = repo.remotes.origin
                    o.pull()
                    print("The plugin repository pulled successfully")
                    return True
                except:
                    print("The plugin repository can not be pulled")
                    return False
            else:
                return False

        else:
            try:
                Repo.clone_from(self.dic_git_url[plu_name], target)
                print('Target directory %s was successfully cloned.' % str(target))
                return True
            except:
                print("The plugin repository can not be cloned")
                return False

    def add_plugin(self, plugin_name):

        if plugin_name == 'UAI':
            answer = self.confirm(f'{self.BOLD}\n This plugin is not obeyed the general PiSCAT license, and it has its specific {self.ENDC}'
                                  f'{self.BOLD}license. please visit "https://github.com/SandoghdarLab/PiSCAT_plugin" for more {self.ENDC}'
                                  f'{self.BOLD}information.{self.ENDC} {self.OKGREEN}\nI agree to install this plugin with a new License (y/n):{self.ENDC}')
            if answer:
                files_changes = self.copy_plugin(plugin_name)
                if files_changes:
                    folder_path = os.path.dirname(os.path.realpath(__file__))
                    path = os.path.join(folder_path, plugin_name)
                    # implement pip as a subprocess:
                    subprocess.run('pip install -r' + path + '\\requirements.txt')

    # def start_tutorials():
    #     """Copy the PiSCAT tutorials to the target directory and start Jupyter server
    #
    #     The target directory is the first argument of sys.argv.
    #     """
    #
    #     target_root = parse_args()
    #     root = copy_tutorials(target_root)
    #     jupyter = find_jupyter()
    #     start_server(root, jupyter)


def main(p_name):
    p_ = Plugin()
    p_.add_plugin(p_name)


if __name__ == '__main__':
    plugin_name = parse_args()
    main(plugin_name)

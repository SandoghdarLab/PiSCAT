# Contributing

Contributions to PiSCAT are always welcome, and they are greatly appreciated!
A list of open problems can be found [here]( https://github.com/SandoghdarLab/PiSCAT/issues).
Of course, it is also always appreciated to bring own ideas and problems to the community!


Please submit all contributions to the official [Github repository](https://github.com/SandoghdarLab/PiSCAT/) in the form of a Merge Request. Please do not submit git diffs or files containing the changes.

`PiSCAT` is an open-source python package under the license of [GNUv3](https://github.com/SandoghdarLab/PiSCAT/blob/main/LICENSE). Thus we consider the act of contributing to the code by submitting a Merge Request as the "Sign off" or agreement to the GNUv3 license.

You can contribute in many different ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/SandoghdarLab/PiSCAT/issues.

### Fix Issues

Look through the Github issues. Different tags are indicating the status of the issues.
The "bug" tag indicates problems with PiSCAT, while the "enhancement" tag shows ideas that should be added in the future.

### Write Documentation

The documentation of PiSCAT can be found [here](https://piscat.readthedocs.io/). [Jupyter notebooks](https://github.com/SandoghdarLab/PiSCAT/tree/main/piscat/Tutorials/JupyterFiles) and [GUI](https://github.com/SandoghdarLab/PiSCAT/tree/main/piscat/GUI) are used to provide an
interactive start to PiSCAT. It is always appreciated if new document notebooks are provided
since this helps others a lot.

## Get Started!

Ready to contribute? Here is how to set up `PiSCAT` for local development.

1. Fork the `PiSCAT` repo on GitHub.
2. Clone your fork locally:
```bash
    $ git clone https://github.com/USERNAME/PiSCAT.git
    $ cd PiSCAT
```
3. Install your local copy into a virtualenv.
```bash
    $ pip install virtualenvwrapper-win (windows)/ pip install virtualenvwrapper (linux)
    $ mkvirtualenv PiSCAT
    $ pip install -e .
```
4. Create a branch for local development:
```bash
    $ git checkout -b name-of-your-bugfix-or-feature
```
   Now you can make your changes locally.

   To get all packages needed for development, a requirements list can be found [here](https://github.com/SandoghdarLab/PiSCAT/blob/main/setup.py).

5. Commit your changes and push your branch to GitHub::
```bash
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
```
6. Submit a Merge Request on Github.

## Merge Request Guidelines

Before you submit a Merge Request, check that it meets these guidelines:

1. All functionality that is implemented through this Merge Request should be covered by unit tests. These are implemented in `PiSCAT\tests`. It would be necessary to add your unit test if the new implementation has some features that are not covered by our unit tests.
2. If the Merge Request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring.
3. If you have a maintainer status for `PiSCAT`, you can merge Merge Requests to the master branch. However, every Merge Request needs to be reviewed by another developer. Thus it is not allowed to merge a Merge Request, which is submitted by oneself.


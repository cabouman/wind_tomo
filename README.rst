.. docs-include-ref

wind_tomo
========

..
    Change the number of = to match the number of characters in the project name.

tomographic reconstruction of wind tunnel turbulance data

..
    Include more detailed description here.

Installing
----------
1. *Clone or download the repository:*

    .. code-block::

        git clone git@github.com:cabouman/wind_tomo

2. Install the conda environment and package

    a. Option 1: Clean install from dev_scripts

        *******You can skip all other steps if you do a clean install.******

        To do a clean install, use the command:

        .. code-block::

            cd dev_scripts
            source clean_install_all.sh

    b. Option 2: Manual install

        1. *Create conda environment:*

            Create a new conda environment named ``wind_tomo`` using the following commands:

            .. code-block::

                conda create --name wind_tomo python=3.10
                conda activate wind_tomo
                pip install -r requirements.txt

            Anytime you want to use this package, this ``wind_tomo`` environment should be activated with the following:

            .. code-block::

                conda activate wind_tomo


        2. *Install wind_tomo package:*

            Navigate to the main directory ``wind_tomo/`` and run the following:

            .. code-block::

                pip install .

            To allow editing of the package source while using the package, use

            .. code-block::

                pip install -e .


Running Demo(s)
---------------

Run any of the available demo scripts with something like the following:

    .. code-block::

        python demo/<demo_file>.py


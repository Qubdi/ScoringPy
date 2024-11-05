Installation
============

Python Version Support
######################

We recommend using the latest version of Python. **ScoringPy** is tested on:

- Python 3.7
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12

Dependencies
############

**ScoringPy** has no mandatory dependencies.

Installation Instructions
#########################

If you encounter any issues, please refer to the FAQ or documentation for troubleshooting tips.

PIP (Preferred)
***************

The recommended way to install **Timeline Manager** is via pip. Use the following command:

.. code-block:: bash

    pip install ScoringPy

After installation, **ScoringPy** is ready to use. Refer to the `Doc <https://timeline-manager.readthedocs.io/en/latest/index.html>`_ for further instructions.

Using Another Package Manager
******************************

**ScoringPy** might be available through some Linux package managers, though using pip ensures you get the latest version.

Ubuntu
-------

As of the latest update, **ScoringPy** is not directly available via Ubuntu package managers. We recommend using pip for installation:

.. code-block:: bash

    sudo apt-get update
    sudo apt-get install python3-pip
    pip3 install ScoringPy

Conda (Anaconda)
----------------

**ScoringPy** is not officially published on conda. However, you can create a custom conda environment and install the library via pip:

.. code-block:: bash

    conda create -n timeline-manager python=3.12
    conda activate timeline-manager
    pip install ScoringPy

Install Manually
****************

If you don't have access to a package manager or need more control, you can manually install the library:

1. Clone the `GitHub <https://github.com/Qubdi/ScoringPy>`_.
2. Navigate to the directory:

   .. code-block:: bash

      cd ScoringPy

3. Install the package:

   .. code-block:: bash

      python setup.py install



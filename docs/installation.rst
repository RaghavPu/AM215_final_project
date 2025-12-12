Installation
============

Requirements
------------

- Python 3.10 or higher
- pip package manager

Standard Installation
---------------------

Clone the repository and install:

.. code-block:: bash

   git clone https://github.com/stasahani1/AM215_final_project.git
   cd AM215_final_project
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install package
   pip install -e .

Development Installation
------------------------

For development (includes testing and linting tools):

.. code-block:: bash

   pip install -e ".[dev]"

Reproducible Environment
------------------------

For exact reproducibility of results:

.. code-block:: bash

   pip install -r requirements-lock.txt


# SPR Analysis

This repository provides a Python implementation for analyzing and curve fitting Surface Plasmon Resonance (SPR) data in a binding assay. The code includes outlier interpolation, data smoothing, and fitting of Langmuir and dissociation decay models. It calculates k_obs, k_on, k_off, R_max, and Kd, and exports the results to files. Additionally, it generates plots for visualization.

## Installation

To run this project, you'll need to have Python installed on your system. The code is compatible with Python 3.x versions.

1. Clone the repository to your local machine:

   ```
   git clone https://github.com/soclemys/spr-analysis
   ```

2. Navigate to the project directory:

   ```
   cd spr-analysis
   ```

3. (Optional) It is recommended to create a virtual environment to keep the project dependencies isolated:

   ```
   python3 -m venv env
   source env/bin/activate
   ```

4. Install the required dependencies using pip:

   ```
   pip install -r requirements.txt
   ```

   This will install all the necessary packages listed in the `requirements.txt` file.

## Usage

1. Place your SPR data files in the root directory.

3. Once the installation is complete, you can run the project:

   ```
   python main.py
   ```

The analysis results, including calculated parameters and plots, will be saved in the `output_data` directory.

## License

This project is licensed under the [MIT License](LICENSE). See the `LICENSE` file for more details.

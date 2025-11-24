🧭 Repository Overview
This repository provides guidelines and part of the source code used in:
Ma et al., 2025

Representative demo file outputs for testing the software.


_________________________________________________________________________________________________________________________________________________________


💻 System Requirements

Operating System

Windows 11 — any base or version (tested on Windows 11 v24H2 x64)

GPU Acceleration (Optional but Recommended) For GPU-accelerated processing, install the CUDA 13.0 toolkit and enable CuPy , which is later utilized by the pyclesperanto plugin | https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11

Recommended Hardware (Minimum)

RAM: 64 GB+ DDR4/DDR5 @ 3600 MT/s

GPU: NVIDIA RTX Quadro P4000 / GeForce RTX 2070 or similar (≥ 8 GB VRAM)

CPU: Intel Core i5-8400 or AMD Ryzen 5 2600 (or better)


_________________________________________________________________________________________________________________________________________________________


📦 Package Installation

All required packages are listed in my_current_env_list.txt. To install dependencies: you can recreate the environment from the provided .yml file using Conda:

conda env create -f environment.yml conda activate Microscopy_Analysis_Advanced2 # example env name

Package Install: The Repository contains a complete package list. Please perform a PIP install for the required and imported package versions as indicated in my_current_env_list.txt and the script.


_________________________________________________________________________________________________________________________________________________________


🧰 Source Code Description

This repository includes the following primary components:
Co-localization Python Analysis Script
.CZI to MIP(Maximum Intensity Projection) Fiji Macro 
Co-localization Python Analysis Script Output .xlsx combiner
Demo File for Co-Localization Analaysis


_________________________________________________________________________________________________________________________________________________________


📋 Usage Guidelines Anaconda Navigator | download Anaconda Navigator and install Anaconda3-2025.06- for Windows, for details, follow the instructions on the website https://www.anaconda.com/download

VS Code | Download VS Studio Code (Version:1.105.1) and install for Windows, for details, follow the instructions on the website: https://code.visualstudio.com/download

.YML Package Install | Download the .YML file on your system and place it in separate folder. Open Anaconda prompt and navigate (base) to the folder directory with the CD command.

_________________________________________________________________________________________________________________________________________________________


📦 Setting Up the Environment (.yml File)

Download and Prepare the Environment File
Download the provided .yml file (e.g., environment_Nucleus.yml or environment.yml) to your system.

Place it in a dedicated folder for clarity.

Create the Conda Environment
Open Anaconda Prompt (in base environment).

Navigate to the folder containing the .yml file using:

Install the required packages as described above (YAML or pip). Run command in Anaconda Propmt while in the right directory,


_________________________________________________________________________________________________________________________________________________________


📋 Usage Guidelines

Download the Demo folder that contains .tiff files. The .tiff files are categorized based on figure label. 
Place the downloaded files in an easy accessible folder. 

Set Up the Environment

Install the required packages as described above (YAML or pip).

Use the recommended Python environment (e.g., Microscopy_Analysis_Advanced2).

Run Co-Localization Script.

Download Co_localization_analysis_script.py (main executable).

Open the file in VS Code or PyCharm.

ensure the interpreter is set to the proper Python 3.9 Environment.

Run the script to generate .html output.

The file has to be first opened in VS code or Pycharm with Proper Pytho Interpreter ( Python, 3.9.2).

<img width="3158" height="1906" alt="Screenshot 2025-11-23 234413" src="https://github.com/user-attachments/assets/90481af9-5a28-438f-9268-aa6604bf2038" />


<img width="3433" height="1360" alt="Screenshot 2025-11-23 235432" src="https://github.com/user-attachments/assets/c7213536-5917-4ea0-b3c9-c08cc60dac92" />

<img width="3438" height="1384" alt="Screenshot 2025-11-23 234637" src="https://github.com/user-attachments/assets/36452c19-237a-4d43-b1d5-022ff482ac99" />




<img width="2660" height="1278" alt="A549-D7-MITO-6_SIM²_MAX_Aligned_intensity_scatter_ColocZ" src="https://github.com/user-attachments/assets/123ebc8e-b7b2-42dc-978b-0aebc4d6e44b" />

<img width="2773" height="1278" alt="A549-D7-NUCLEAR-29_SIM²_MAX_Aligned_intensity_scatter_ColocZ" src="https://github.com/user-attachments/assets/3dfc3925-ce0b-4543-99c9-4de565a7b098" />

<img width="2670" height="1278" alt="A549-HD7-ER-meth29_SIM²_MAX_Aligned_intensity_scatter_ColocZ" src="https://github.com/user-attachments/assets/efc6d681-136d-4098-af3e-fd1ed0e60fc2" />

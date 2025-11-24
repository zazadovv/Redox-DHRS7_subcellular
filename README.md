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

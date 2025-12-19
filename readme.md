TODO:

1. Enable AMP (Automatic Mixed Precision) training. 
Ensure that the CUDA versions of the virtual environment and system are consistent.

2. Install Dependencies
Install the required Python packages (including PyTorch with CUDA 11.8 support):

Note: Ensure your system's CUDA version is consistent with the PyTorch version installed.

🚀 Quick Start (快速运行)
We provide an all-in-one script run.sh to launch the entire system automatically.

1. Grant Permissions
Make sure the script is executable:

2. Launch System
Execute the script:

⚙️ What happens when you run this script?
The run.sh automation script performs the following steps sequentially:

Simulation: Launches Gazebo SITL (using base environment).

Initialization: Runs takeoff.py to arm the drone and switch to OFFBOARD mode.

Handover: Once the drone is armed (Vehicle armed), it automatically stops the takeoff script.

Planning: Launches the main planner (yopo_mvpa.py and test4.py) and opens RViz for visualization.

📂 File Structure 
run.sh: The master automation script to launch simulation and planner.

takeoff.py: Handles initial drone arming and offboard switching.

test4.py: Custom testing/planning module.

requirements.txt: List of project dependencies.

config/: Configuration files for trajectory optimization.

🧩 Acknowledgements 
This project is built upon the excellent work of TJU-Aerial-Robotics. We strictly adhere to the open-source spirit and acknowledge their contribution.

Original Repository:https://github.com/TJU-Aerial-Robotics/YOPO
📄 License
This project follows the license of the original YOPO repository.
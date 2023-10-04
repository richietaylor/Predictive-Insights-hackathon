import os
import subprocess

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# List of Python scripts to execute in order
scripts_to_execute = [
    "./1_Code/processor.py",
    "./1_Code/stacking.py"
]

# Execute the scripts using their relative paths
for script in scripts_to_execute:
    script_path = os.path.join(script_dir, script)
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_path}: {e}")
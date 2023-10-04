import subprocess
import threading
import time
import re

output = ""
# Function to execute a Python script
def run_script(script):
    global output
    try:
        # Run the script and capture the output
        result = subprocess.run(
            ["python", script], capture_output=True, text=True, check=True
        )

        score = re.search(r'(Mean.+?0\.\d+)|(MSE.+? 0\.\d+ \[.+\])',result.stdout)
        # print(score)
        # Print a message indicating the successful execution of the script
        if not score:
            score =""
        else:
            score = score.group()
           
        print(f"Script '{script}' executed successfully.\t  " + score)
    except subprocess.CalledProcessError as e:
        # Handle errors that occur during script execution
        print(f"Error executing script '{script}':")
        print(e)
    except Exception as e:
        # Handle other exceptions that may occur during script execution
        print(f"Error executing script '{script}':")
        print(e)


# Function for the loading animation
def loading_animation():
    while not all(script_done_flags):
        for char in "|/-\\":
            if all(script_done_flags):
                break
            print(f"\rExecuting scripts {char} ", end="")
            time.sleep(0.1)


# Define a list of Python scripts to run in order
python_scripts = [
    "./1_Code/processor.py",
    "./1_Code/1_lightgbm.py",
    "./1_Code/2_xgboost.py",
    "./1_Code/3_cboost.py",
    "./1_Code/5_randomForest.py",
    "./1_Code/6_histboosting.py",
    "./1_Code/merge.py",
]

# Create a list of flags to track script execution status
script_done_flags = [False] * len(python_scripts)

# Start the loading animation in a separate thread
animation_thread = threading.Thread(target=loading_animation)
animation_thread.start()

# Loop through the list of Python scripts and execute them one by one
for i, script in enumerate(python_scripts):
    run_script(script)
    script_done_flags[i] = True

# Wait for the loading animation thread to finish
animation_thread.join()

# Print a message indicating the completion of all scripts
print("\nAll scripts have been executed.")


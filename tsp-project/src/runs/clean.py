import os

# Define the directory path
directory_path = "/Users/gabrielpecoraro/Desktop/Cours_ENSEIRB/3A/IIT/Spring 2025/Research/research_project/tsp-project/src/runs/tsp_training"

# Check if the directory exists
if os.path.exists(directory_path):
    try:
        # List all files in the directory
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            # Remove each file
            os.remove(file_path)
        print(f"All files in directory {directory_path} have been removed.")
    except PermissionError:
        print(f"Permission denied to remove files in directory {directory_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print(f"Directory {directory_path} does not exist.")

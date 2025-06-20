import os
import argparse


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Clean training files and images")
    parser.add_argument(
        "--images",
        choices=["on", "off"],
        default="off",
        help="Whether to remove PNG files in training_progress directory (on/off)",
    )
    args = parser.parse_args()

    # Define the directory paths
    directory_path = "/Users/gabrielpecoraro/Desktop/Cours_ENSEIRB/3A/IIT/Spring 2025/Research/research_project/tsp-project/src/runs/tsp_training"
    directory_path2 = "/Users/gabrielpecoraro/Desktop/Cours_ENSEIRB/3A/IIT/Spring 2025/Research/research_project/tsp-project/src/training_progress"

    # Clean TensorBoard logs
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

    # Clean training progress images if requested
    if args.images == "on" and os.path.exists(directory_path2):
        try:
            # List all files in the directory
            files = os.listdir(directory_path2)
            removed_count = 0
            for file in files:
                if file.lower().endswith(".png"):  # Only remove PNG files
                    file_path = os.path.join(directory_path2, file)
                    os.remove(file_path)
                    removed_count += 1
            print(
                f"Removed {removed_count} PNG files from directory {directory_path2}."
            )
        except PermissionError:
            print(f"Permission denied to remove files in directory {directory_path2}.")
        except Exception as e:
            print(f"An error occurred while cleaning images: {e}")
    elif args.images == "on":
        print(f"Directory {directory_path2} does not exist.")


if __name__ == "__main__":
    main()

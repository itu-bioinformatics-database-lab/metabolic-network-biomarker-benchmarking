import subprocess
import os

# Define the steps and their corresponding script paths
pipeline_steps = [
    "extract_and_sort_scores.py",
    "refmet_mapper.py",
    "common_metabolite_finder.py",
    "preprocess_training.py",
    "train.py"
]


def run_script(script_name):
    print(f"\nRunning: {script_name}")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
        exit(1)
    else:
        print(result.stdout)


if __name__ == "__main__":
    for step in pipeline_steps:
        run_script(step)
    print("\nPipeline completed successfully.")

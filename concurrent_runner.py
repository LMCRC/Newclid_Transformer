import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


# Function to run a command
def run_command(command: str):
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        return (command, result.stdout)
    except subprocess.CalledProcessError as e:
        return (command, e.stderr)


# List of commands to run simultaneously
commands = [
    "alphageo --device cpu -o p5 --problems-file problems_datasets/examples.txt --problem translated_imo_2018_p1 --search-width 5 --search-depth 2 --batch-size 5 --lm-beam-width 5",
    "alphageo --device cpu -o p15 --problems-file problems_datasets/examples.txt --problem translated_imo_2018_p1 --search-width 15 --search-depth 2 --batch-size 15 --lm-beam-width 15",
    "alphageo --device cpu -o p55 --problems-file problems_datasets/examples.txt --problem translated_imo_2018_p1 --search-width 55 --search-depth 2 --batch-size 32 --lm-beam-width 32",
    "alphageo --device cpu -o p512 --problems-file problems_datasets/examples.txt --problem translated_imo_2018_p1 --search-width 512 --search-depth 16 --batch-size 32 --lm-beam-width 32",
]

# Using ThreadPoolExecutor to run commands in parallel
with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    # Start the command execution
    futures = [executor.submit(run_command, cmd) for cmd in commands]

    # Collect the results as they complete
    for future in as_completed(futures):
        command, output = future.result()
        print(f"Command: {command}\nOutput:\n{output}\n{'='*60}")

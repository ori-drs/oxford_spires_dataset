import subprocess
from pathlib import Path


class TerminalColors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    ORANGE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_with_colour(text, colour=TerminalColors.CYAN):
    print(f"{colour}{text}{TerminalColors.ENDC}")


def run_command(cmd, log_path=None, print_command=True):
    if print_command:
        print_with_colour(f"Running command: {cmd}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, universal_newlines=True)
    for line in process.stdout:
        print(line, end="")
        if log_path is not None:
            assert isinstance(log_path, (Path, str))
            with open(log_path, "a") as f:
                f.write(line)

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


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
    logger.info(f"{colour}{text}{TerminalColors.ENDC}")


def run_command(cmd, log_path=None, print_command=True, print_output=True):
    if print_command:
        print_with_colour(f"Running command: {cmd}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, universal_newlines=True)
    if print_output:
        for line in process.stdout:
            logger.info(line.rstrip())
            if log_path is not None:
                assert isinstance(log_path, (Path, str))
                with open(log_path, "a") as f:
                    f.write(line)
    else:
        return process

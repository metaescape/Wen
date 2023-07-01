import argparse
import os
import subprocess
import sys
from wen.configuration import WenConfig

CFG = WenConfig()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("command", choices=["demo", "init"])
args = parser.parse_args()

# Check if the command is "demo"
if args.command == "demo":
    # Change to the directory where the script is located
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Check if the GodTian_Pinyin directory exists
    if not os.path.exists("GodTian_Pinyin"):
        # If not, clone the repository
        subprocess.run(
            ["git", "clone", "git@github.com:whatbeg/GodTian_Pinyin.git"],
            check=True,
        )

    # Change to the GodTian_Pinyin directory
    os.chdir("GodTian_Pinyin")

    # Apply the patch
    subprocess.run(["git", "apply", "../0001-for-py3.patch"], check=True)

# Check if the command is "init"
elif args.command == "init":
    # Create a script in ~/.local/bin/
    home_dir = os.path.expanduser("~")
    script_path = os.path.join(home_dir, ".local", "bin", "wenls")
    proj_dir = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(script_path):
        print(f"Script {script_path} already exists")
        print(f"please rm {script_path} first")
        sys.exit(1)
    with open(script_path, "w") as script_file:
        script_file.write(
            f"""\
#!{CFG.python_path}
import sys
sys.path.append('{os.path.join(proj_dir, "wen")}')
import os
os.chdir("{os.path.join(proj_dir, "wen")}")
from server import wenls
wenls.start_io()
"""
        )
    # Make the script executable
    os.chmod(script_path, 0o755)

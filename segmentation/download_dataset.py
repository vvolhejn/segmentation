import argparse
from pathlib import Path
import subprocess

FILE_LIST_URL = "https://scontent.fprg4-1.fna.fbcdn.net/m1/v/t6/An8MNcSV8eixKBYJ2kyw6sfPh-J9U4tH2BV7uPzibNa0pu4uHi6fyXdlbADVO4nfvsWpTwR8B0usCARHTz33cBQNrC0kWZsD1MbBWjw.txt?ccb=10-5&oh=00_AfC8bPvgvIxtx56j_bM_fKaZS1JyPGgRHoF41GqBAonIOg&oe=65A97418&_nc_sid=0fdd51"

REPO_ROOT = Path(__file__).parents[1]
SA_1B_DIR = REPO_ROOT / "data" / "sa_1b"
FILE_LIST_FILENAME = "file_list.txt"


def download_file_list():
    """Downloads the file that contains the links to the actual .tar files.

    This file is generated dynamically on their side so re-downloading is necessary.
    """
    SA_1B_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["wget", "-O", FILE_LIST_FILENAME, FILE_LIST_URL], cwd=SA_1B_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    args = parser.parse_args()

    download_file_list()

    with open(SA_1B_DIR / FILE_LIST_FILENAME, "r") as f:
        name_to_url = {
            k: v for k, v in [line.strip().split("\t") for line in f.readlines()][1:]
        }

    if not args.file:
        print(name_to_url.keys())
        print("--file not given, please select one of the above")
        exit(1)

    if args.file not in name_to_url:
        print(name_to_url.keys())
        print(
            f"File {args.file} not found in the list. Please select one of the above."
        )
        exit(1)

    (SA_1B_DIR / "compressed").mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["wget", "-O", args.file, name_to_url[args.file]],
        cwd=SA_1B_DIR / "compressed",
    )

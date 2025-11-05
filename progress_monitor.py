import sys
import time


def progress_bar(total, current):
    bar_length = 50  # progress bar length (in symbols)
    fraction = current / total
    filled_length = int(bar_length * fraction)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percent = int(100 * fraction)
    sys.stdout.write(f'\r|{bar}| {percent}% ({current}/{total})')
    sys.stdout.flush()
    if current == total:
        print()  # Newstring at the end

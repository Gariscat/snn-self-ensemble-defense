import re

def compare_files_ignore_device(file1_path, file2_path):
    """
    Compare two text files line by line, ignoring device-specific differences (e.g., f"cuda:{x}" vs "cuda:x").

    Parameters:
    - file1_path: Path to the first text file.
    - file2_path: Path to the second text file.
    """
    def normalize_line(line):
        """Normalize a line to ignore device-specific details."""
        return re.sub(r'cuda:\d', 'cuda:x', line.strip())

    try:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
            lines1 = file1.readlines()
            lines2 = file2.readlines()

        max_lines = max(len(lines1), len(lines2))
        print(f"Comparing '{file1_path}' and '{file2_path}':\n")

        for i in range(max_lines):
            line1 = normalize_line(lines1[i]) if i < len(lines1) else "<No Line>"
            line2 = normalize_line(lines2[i]) if i < len(lines2) else "<No Line>"

            if line1 != line2:
                print(f"Line {i + 1}:")
                print(f"  File1: {line1}")
                print(f"  File2: {line2}\n")
                break

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def compare_files(file1_path, file2_path):
    """
    Compare two text files line by line and display the differences.

    Parameters:
    - file1_path: Path to the first text file.
    - file2_path: Path to the second text file.
    """
    try:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
            lines1 = file1.readlines()
            lines2 = file2.readlines()

        max_lines = max(len(lines1), len(lines2))
        print(f"Comparing '{file1_path}' and '{file2_path}':\n")

        for i in range(max_lines):
            line1 = lines1[i].strip() if i < len(lines1) else "<No Line>"
            line2 = lines2[i].strip() if i < len(lines2) else "<No Line>"
            
            if line1 != line2:
                print(f"Line {i + 1}:")
                print(f"  File1: {line1}")
                print(f"  File2: {line2}\n")
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
# compare_files("file1.txt", "file2.txt")
compare_files_ignore_device("alpha_gpu_0.txt", "alpha_gpu_3.txt")
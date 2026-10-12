import os

# Output file name
output_file = "full_project_code.txt"

# Large binary or cache folders to skip
ignore_dirs = {
    ".git",
    "__pycache__",
    "cache",
    ".dist",
    "models",
    "predictions",
}

# File types you want to include
valid_extensions = {".py", ".txt", ".json", ".md", ".yaml", ".yml"}

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk("."):
        # Skip specified directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for file in files:
            if file == output_file or file == "combine_files.py":
                continue

            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                filepath = os.path.join(root, file)

                # Write section header with file path
                outfile.write(f"\n{'='*60}\n")
                outfile.write(f"FILE: {filepath}\n")
                outfile.write(f"{'='*60}\n\n")

                try:
                    with open(filepath, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n")
                except Exception as e:
                    outfile.write(f"[Error reading file: {e}]\n")

print(
    f"Successfully created '{output_file}'! You can now upload this file to Copilot."
)
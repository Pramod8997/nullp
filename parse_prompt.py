import re

with open("nullp_exhaustive_test_prompt.md", "r") as f:
    lines = f.readlines()

sections = []
current_file = None
current_content = []

for line in lines:
    file_match = re.search(r'\*\*File:\*\*\s*`([^`]+)`', line)
    if file_match:
        if current_file:
            sections.append((current_file, "".join(current_content)))
        current_file = file_match.group(1)
        current_content = [line]
    elif current_file:
        if line.startswith("### ") and not "**File:**" in "".join(lines[lines.index(line):lines.index(line)+2]):
            # If we hit a new section that doesn't have a file, wait it out
            pass
        current_content.append(line)

if current_file:
    sections.append((current_file, "".join(current_content)))

import json
with open("test_requirements.json", "w") as f:
    json.dump(sections, f, indent=2)

print(f"Extracted {len(sections)} files to test.")
for f, c in sections:
    print(f)

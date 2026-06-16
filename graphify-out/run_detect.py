import sys
import json
import os
from pathlib import Path
from graphify.detect import detect

def main():
    # Step 1: Initialize
    print("Initializing...")
    python_path = sys.executable
    root_path = os.getcwd()
    
    Path('graphify-out/.graphify_python').write_text(python_path, encoding='utf-8')
    Path('graphify-out/.graphify_root').write_text(root_path, encoding='utf-8')
    print(f"Set interpreter: {python_path}")
    print(f"Set root: {root_path}")
    
    # Step 2: Detect
    print("Scanning directory...")
    result = detect(Path('.'))
    
    # Filter out files inside graphify-out
    # Note: detect already skips graphify-out
    
    Path('graphify-out/.graphify_detect.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    
    total_files = result.get('total_files', 0)
    total_words = result.get('total_words', 0)
    print(f"\nCorpus: {total_files} files · ~{total_words} words")
    for k, v in result.get('files', {}).items():
        if v:
            print(f"  {k}: {len(v)} files")
            
    if result.get('skipped_sensitive'):
        print(f"  (Skipped {len(result['skipped_sensitive'])} sensitive files)")

if __name__ == '__main__':
    main()

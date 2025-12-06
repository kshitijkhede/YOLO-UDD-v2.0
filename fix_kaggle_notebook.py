#!/usr/bin/env python3
"""
Fix YOLO-UDD Kaggle Notebook - Apply all critical corrections
"""

import json
import sys

def fix_notebook(input_file, output_file):
    """Apply all critical fixes to the Kaggle notebook"""
    
    print(f"📖 Reading notebook: {input_file}")
    with open(input_file, 'r') as f:
        notebook = json.load(f)
    
    fixes_applied = []
    
    # Process each cell
    for cell in notebook.get('cells', []):
        source_lines = cell.get('source', [])
        source = ''.join(source_lines)
        original_source = source
        
        # Fix #1: Training parameters
        if "'epochs':" in source and ("'epochs': 20" in source or "'epochs': 50" in source):
            source = source.replace("'epochs': 20,", "'epochs': 100,  # PDF spec: 300 (100 for initial training)")
            source = source.replace("'epochs': 50,", "'epochs': 100,  # PDF spec: 300 (100 for initial training)")
            fixes_applied.append("Fixed epochs: 20/50 -> 100")
        
        if "'batch_size':" in source and ("'batch_size': 4" in source or "'batch_size': 8" in source):
            source = source.replace("'batch_size': 4,", "'batch_size': 16,  # PDF spec: 16")
            source = source.replace("'batch_size': 8,", "'batch_size': 16,  # PDF spec: 16")
            fixes_applied.append("Fixed batch_size: 4/8 -> 16")
        
        if "'learning_rate': 0.001" in source:
            source = source.replace("'learning_rate': 0.001,", "'learning_rate': 0.01,  # PDF spec: 0.01")
            fixes_applied.append("Fixed learning_rate: 0.001 -> 0.01")
        
        # Fix #2: Checkpoint extensions .pth -> .pt
        if ".pth" in source and "glob.glob" in source:
            source = source.replace("*.pth')", "*.pt')  # Fixed: .pt not .pth")
            source = source.replace("best.pth')", "best.pt')  # Fixed: .pt not .pth")
            fixes_applied.append("Fixed checkpoint extensions: .pth -> .pt")
        
        # Apply changes if source was modified
        if source != original_source:
            # Split back into lines preserving original line breaks
            cell['source'] = source.splitlines(keepends=True)
    
    print(f"\n✅ Fixes applied ({len(fixes_applied)}):")
    for fix in set(fixes_applied):
        print(f"   - {fix}")
    
    print(f"\n💾 Writing fixed notebook: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print("✅ Done!")

if __name__ == "__main__":
    input_file = "YOLO_UDD_Kaggle_Training_AutoResume.ipynb"
    output_file = "YOLO_UDD_Kaggle_Training_AutoResume_FIXED.ipynb"
    
    try:
        fix_notebook(input_file, output_file)
        print(f"\n📝 Next steps:")
        print(f"   1. Review: {output_file}")
        print(f"   2. Test locally (optional)")
        print(f"   3. Upload to Kaggle")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

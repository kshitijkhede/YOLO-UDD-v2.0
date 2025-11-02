"""Find duplicate files by SHA-1 hash and print groups of identical files.
Excludes .git, weights, and typical venv directories.
"""
import os, hashlib, sys
root='.'
exclude_dirs={'./.git','./weights','./.venv','./venv','./__pycache__'}
files_by_hash={}
for dirpath,dirnames,filenames in os.walk(root):
    nd = os.path.normpath(dirpath)
    if any(nd==e or nd.startswith(e+os.sep) for e in exclude_dirs):
        continue
    for f in filenames:
        path=os.path.join(dirpath,f)
        # skip inside excluded paths
        if path.startswith('./.git') or path.startswith('./weights'):
            continue
        try:
            with open(path,'rb') as fh:
                h=hashlib.sha1(fh.read()).hexdigest()
        except Exception as e:
            print("SKIP %s: %s" % (path, e), file=sys.stderr)
            continue
        files_by_hash.setdefault(h,[]).append(path)

found=False
for h,paths in sorted(files_by_hash.items(), key=lambda x: -len(x[1])):
    if len(paths)>1:
        found=True
        print('--- GROUP (count=%d) ---' % len(paths))
        for p in paths:
            print(p)
if not found:
    print('No duplicates found')

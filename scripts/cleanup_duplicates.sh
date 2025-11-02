#!/usr/bin/env bash
# scripts/cleanup_duplicates.sh
# Robust cleanup helper: computes duplicates via a temp Python helper file,
# supports --yes for non-interactive runs, removes duplicates and common
# temporary files, commits and (optionally) pushes.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./scripts/cleanup_duplicates.sh [--yes] [--no-push]

Options:
  --yes       Run non-interactively (auto-confirm deletions).
  --no-push   Do not push the cleanup commit to origin/main.
  -h,--help   Show this help.
USAGE
}

AUTO_YES=0
NO_PUSH=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --yes) AUTO_YES=1; shift ;;
    --no-push) NO_PUSH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

echo "Scanning repository for duplicate files and common temporary files..."
echo

# Use a temporary python script to compute duplicate files to delete (keep first)
TMP_PY=$(mktemp --suffix=_dupes.py)
cat > "$TMP_PY" <<'PY'
import os,hashlib
exclude_dirs={'./.git','./weights','./.venv','./venv','./__pycache__'}
files_by_hash={}
for dirpath,dirnames,filenames in os.walk('.'):
    nd=os.path.normpath(dirpath)
    if any(nd==e or nd.startswith(e+os.sep) for e in exclude_dirs):
        continue
    for f in filenames:
        path=os.path.join(dirpath,f)
        if path.startswith('./.git') or path.startswith('./weights'):
            continue
        try:
            with open(path,'rb') as fh:
                h=hashlib.sha1(fh.read()).hexdigest()
        except Exception:
            continue
        files_by_hash.setdefault(h,[]).append(path)
del_list=[]
for h,paths in files_by_hash.items():
    if len(paths)>1:
        paths.sort()
        # keep first, delete rest
        del_list.extend(paths[1:])
for p in sorted(del_list):
    print(p)
PY

dup_delete_list=$(python3 "$TMP_PY" || true)
rm -f "$TMP_PY"

if [ -z "$dup_delete_list" ]; then
  echo "No duplicate files to delete."
else
  echo "Duplicate files that would be deleted (keeping the first file of each group):"
  echo "$dup_delete_list"
fi
echo

# 3) Find common temporary files
tmp_files=$(find . -type f \( -name '*.pyc' -o -name '*~' -o -name '.DS_Store' -o -path '*/__pycache__/*' -o -path '*/.ipynb_checkpoints/*' \) -print || true)
if [ -z "$tmp_files" ]; then
  echo "No common temporary files (.pyc, __pycache__, .DS_Store, ipynb checkpoints) found."
else
  echo "Temporary files / bytecode files found:"
  echo "$tmp_files"
fi
echo

if [ "$AUTO_YES" -eq 0 ]; then
  echo "WARNING: The operations below will permanently delete files from disk and remove tracked files from git."
  read -r -p "Do you want to proceed and delete the listed files and cleanup caches? Type 'yes' to proceed: " confirm
  if [ "$confirm" != "yes" ]; then
    echo "Aborted by user. No changes made."
    exit 0
  fi
else
  echo "Auto-confirm enabled (--yes): proceeding without interactive prompt."
fi

# Delete duplicate files
if [ -n "$dup_delete_list" ]; then
  echo "Removing duplicate files..."
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    if git ls-files --error-unmatch "$f" > /dev/null 2>&1; then
      git rm -f "$f" || true
    else
      rm -f "$f" || true
    fi
  done <<< "$dup_delete_list"
fi

# Remove temporary files and __pycache__ directories
echo "Removing temporary files and __pycache__ directories..."
while IFS= read -r f; do
  [ -z "$f" ] && continue
  if git ls-files --error-unmatch "$f" > /dev/null 2>&1; then
    git rm -f "$f" || true
  else
    rm -f "$f" || true
  fi
done <<< "$tmp_files"

find . -type d -name '__pycache__' -print -exec rm -rf {} + || true
find . -type d -path '*/.ipynb_checkpoints' -print -exec rm -rf {} + || true

# Stage, commit, and optionally push
git add -A
if ! git diff --cached --quiet; then
  git commit -m "Cleanup: remove duplicate & temporary files"
  if [ "$NO_PUSH" -eq 0 ]; then
    echo "Pushing cleanup commit to origin main..."
    git push origin main
  else
    echo "--no-push specified; skipping push."
  fi
else
  echo "No changes to commit after cleanup."
fi

echo "Cleanup complete."


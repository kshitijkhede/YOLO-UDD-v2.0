#!/bin/bash
# Quick sync script to push local changes to GitHub

echo "📝 Checking git status..."
git status

echo ""
echo "➕ Staging all changes..."
git add -A

echo ""
echo "📊 Changes to be committed:"
git status --short

echo ""
read -p "Enter commit message (or press Enter for default): " commit_msg

if [ -z "$commit_msg" ]; then
    commit_msg="Update project files - $(date '+%Y-%m-%d %H:%M:%S')"
fi

echo ""
echo "💾 Committing changes..."
git commit -m "$commit_msg"

echo ""
echo "🚀 Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Sync complete!"

#!/bin/bash
# Push YOLO-UDD v2.0 to GitHub
# 
# Usage: ./push_to_github.sh YOUR_GITHUB_USERNAME

if [ -z "$1" ]; then
    echo "❌ Error: GitHub username required"
    echo ""
    echo "Usage: ./push_to_github.sh YOUR_GITHUB_USERNAME"
    echo ""
    echo "Example: ./push_to_github.sh john_doe"
    exit 1
fi

USERNAME=$1
REPO_URL="https://github.com/${USERNAME}/YOLO-UDD-v2.0.git"

echo "=========================================="
echo "  Pushing to GitHub"
echo "=========================================="
echo "Repository: $REPO_URL"
echo ""

# Check if remote already exists
if git remote get-url origin 2>/dev/null; then
    echo "ℹ️  Remote 'origin' already exists. Removing it first..."
    git remote remove origin
fi

# Add remote
echo "📡 Adding remote..."
git remote add origin "$REPO_URL"

# Verify remote
echo ""
echo "✅ Remote added:"
git remote -v

echo ""
echo "=========================================="
echo "  Ready to Push"
echo "=========================================="
echo ""
echo "Run this command to push:"
echo ""
echo "  git push -u origin main"
echo ""
echo "When prompted:"
echo "  Username: $USERNAME"
echo "  Password: Use your Personal Access Token"
echo ""
echo "=========================================="

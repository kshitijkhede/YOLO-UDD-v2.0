# 🚀 GitHub Setup Instructions

## ✅ Git Repository Initialized!

Your project is now a git repository with the initial commit made.

---

## 📤 Push to GitHub - Step by Step

### Step 1: Create GitHub Repository

1. Go to: https://github.com/new
2. Repository name: `YOLO-UDD-v2.0`
3. Description: `Turbidity-Adaptive Architecture for Underwater Debris Detection`
4. Keep it **Public** or **Private** (your choice)
5. **DO NOT** initialize with README, .gitignore, or license (we already have them)
6. Click "Create repository"

### Step 2: Copy Your GitHub Repository URL

After creating, GitHub will show you a URL like:
- HTTPS: `https://github.com/YOUR_USERNAME/YOLO-UDD-v2.0.git`
- SSH: `git@github.com:YOUR_USERNAME/YOLO-UDD-v2.0.git`

### Step 3: Connect and Push

Run these commands (replace YOUR_USERNAME with your actual GitHub username):

```bash
cd /home/student/MIR/Project/YOLO-UDD-v2.0

# Add GitHub as remote (use HTTPS)
git remote add origin https://github.com/YOUR_USERNAME/YOLO-UDD-v2.0.git

# Push to GitHub
git push -u origin main
```

OR if using SSH:

```bash
git remote add origin git@github.com:YOUR_USERNAME/YOLO-UDD-v2.0.git
git push -u origin main
```

---

## 🔑 GitHub Authentication

### For HTTPS:
You'll need a Personal Access Token (PAT):

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (all)
4. Copy the token
5. Use it as password when pushing

### For SSH:
Set up SSH keys:

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: https://github.com/settings/ssh/new
```

---

## 📋 Quick Command Reference

```bash
# Check git status
git status

# View commit history
git log --oneline

# Add more changes
git add .
git commit -m "Your commit message"
git push

# Create and switch to new branch
git checkout -b feature-branch

# View remote URL
git remote -v
```

---

## 🔄 Making Updates

After making changes to your code:

```bash
# Stage changes
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push
```

---

## 📁 What's Included in Git

✅ **Included:**
- All source code (models/, scripts/, utils/)
- Configuration files
- Documentation (README, etc.)
- Requirements.txt
- Setup script

❌ **Excluded (in .gitignore):**
- Virtual environment (venv/)
- Dataset images (large files)
- Training outputs (runs/)
- Model checkpoints (*.pt, *.pth)
- Python cache (__pycache__)

---

## 💡 Tips

1. **Large Files**: Dataset is excluded. Document where to download it in README
2. **Model Weights**: Can be shared via Google Drive or GitHub Releases
3. **Branches**: Use branches for experiments (git checkout -b experiment-name)
4. **Commits**: Make frequent, descriptive commits
5. **README**: Update README with your results and findings

---

## 🆘 Common Issues

**Authentication failed?**
- Use Personal Access Token instead of password
- Or set up SSH keys

**Large files rejected?**
- Check .gitignore is working
- Use Git LFS for large files: `git lfs install`

**Remote already exists?**
```bash
git remote remove origin
git remote add origin YOUR_NEW_URL
```


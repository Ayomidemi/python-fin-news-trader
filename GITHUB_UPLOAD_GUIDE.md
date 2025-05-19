# How to Upload This Project to GitHub

Follow these steps to upload the FinNews Trader project to your GitHub account:

## 1. Download the Code from Replit

First, you need to download all the project files from Replit:

1. In your Replit project, click on the three dots menu in the file browser panel
2. Select "Download as zip"
3. Save the ZIP file to your computer and extract it

## 2. Create a New GitHub Repository

1. Go to [GitHub](https://github.com/) and log in to your account
2. Click the "+" icon in the top right corner and select "New repository"
3. Name your repository (e.g., "fin-news-trader")
4. Add an optional description
5. Choose public or private visibility
6. Click "Create repository"

## 3. Upload the Files to GitHub

### Option 1: Using GitHub Web Interface (Easiest)

1. In your new GitHub repository, click on "uploading an existing file" link
2. Drag and drop all the extracted files from the Replit project
3. Add a commit message (e.g., "Initial commit")
4. Click "Commit changes"

### Option 2: Using Git Command Line (Advanced)

If you're familiar with Git:

1. Open a terminal/command prompt
2. Navigate to the directory where you extracted the Replit files
3. Run the following commands:

```bash
# Initialize a new git repository
git init

# Add all files to the repository
git add .

# Create your first commit
git commit -m "Initial commit"

# Connect to your GitHub repository (replace with your actual repository URL)
git remote add origin https://github.com/YOUR_USERNAME/fin-news-trader.git

# Push files to GitHub
git push -u origin main
```

## 4. Add Dependencies Information

After uploading, you might want to add a requirements.txt file to list dependencies. Here are the requirements for this project:

- streamlit
- pandas
- numpy
- plotly
- yfinance
- nltk
- spacy
- trafilatura
- beautifulsoup4
- tqdm
- requests

## 5. Installing Dependencies

For anyone who wants to use your code, they should:

1. Clone your repository
2. Install Python (3.9 or higher recommended)
3. Install dependencies using pip:
   ```
   pip install streamlit pandas numpy plotly yfinance nltk spacy trafilatura beautifulsoup4 tqdm requests
   ```
4. Download the required spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```
5. Run the app:
   ```
   streamlit run app.py
   ```
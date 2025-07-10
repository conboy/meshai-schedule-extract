# Schedule Extract Tool

Script to extract shift information from PDF or PNG files using OpenRouter API.

## Setup

### Option 1: Automatic Setup (Recommended)
Run the setup script:
```bash
./setup.sh
```

This will:
- Install uv (if not already installed)
- Create a virtual environment
- Install all dependencies

### Option 2: Manual Setup
1. Install uv if not already installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml
```

3. Set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Usage

Make sure your virtual environment is activated:
```bash
source .venv/bin/activate
```

Run the script:
```bash
python extract_shifts.py <file_path>
```

Example:
```bash
python extract_shifts.py "data/New Schedule 1.pdf"
```

Or run all files in the data directory:
```bash
python main.py
```

## Supported Formats
- PDF files (converted to PNG)
- PNG images
- JPG/JPEG images

## Output

The script will:
1. Send the file to OpenRouter for analysis
2. Extract all shifts with fields: Start Date, Start Time, End Date, End Time, Job, Member, Worked Hours, Note
3. Save results as JSON (or text if JSON parsing fails)
4. Display the extracted shifts

## Package Management

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package management.

### Adding new dependencies:
```bash
uv add <package-name>
```

### Removing dependencies:
```bash
uv remove <package-name>
```

### Updating dependencies:
```bash
uv pip install --upgrade -r requirements.txt
```

## Available Files

The `data/` directory contains various schedule files you can test with:
- PDF schedules
- PNG images
- Excel files (not supported by this script)

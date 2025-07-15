# Unstructured Schedule Extraction Agents

Agents built for meshai.io to extract employee shift information from unstructured schedule files.

## Setup

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
- PDF
- PNG
- Excel
- Word Documents

## TODO: Refactor Structure

The repo needs a major refactor to improve organization and maintainability:

### Proposed Structure:
```
main.py
config.py
logging.py

data/

file_conversion/
├── doc_to_pdf.py
├── doc_to_csv.py (convert tables inside doc to csv)
├── excel_to_png.py (using libreoffice)
├── excel_to_csv.py
└── pdf_to_png.py

agents/
├── decision_agent.py
├── png_agent.py
├── csv_agent.py
├── code analysis agent (generates python script to extract data directly from excel file)
├── png_csv_agent.py (benchmark against separate agents)
└── post_processing_agent.py (role TBD)

testing/
├── input_dataset/
├── output_dataset/
└── compare_script.py
```

### Refactor Tasks:
- [ ] Restructure files according to proposed organization
- [ ] Create modular file conversion utilities
- [ ] Implement specialized agents for different data types
- [ ] Set up proper testing framework with validation
- [ ] Add comprehensive unit tests for all modules
- [ ] Benchmark combined vs separate agents
- [ ] Add proper configuration management
- [ ] Implement centralized logging

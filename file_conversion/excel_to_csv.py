import os
import pandas as pd
from pathlib import Path
import logging


def convert_excel_to_csv(input_dir: str, output_dir: str) -> None:
    """
    Convert all Excel files in input directory to CSV format in output directory.
    
    Args:
        input_dir: Path to directory containing Excel files
        output_dir: Path to directory where CSV files will be saved
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    excel_extensions = ['.xlsx', '.xls', '.xlsm']
    excel_files = []
    
    for ext in excel_extensions:
        excel_files.extend(input_path.glob(f"*{ext}"))
    
    if not excel_files:
        logging.warning(f"No Excel files found in {input_dir}")
        return
    
    for excel_file in excel_files:
        try:
            convert_single_excel_file(excel_file, output_path)
            logging.info(f"Successfully converted {excel_file.name}")
        except Exception as e:
            logging.error(f"Failed to convert {excel_file.name}: {str(e)}")


def convert_single_excel_file(excel_file: Path, output_dir: Path) -> None:
    """
    Convert a single Excel file to CSV format(s).
    
    Args:
        excel_file: Path to the Excel file
        output_dir: Directory to save CSV files
    """
    base_name = excel_file.stem
    
    try:
        xl_file = pd.ExcelFile(excel_file)
        sheet_names = xl_file.sheet_names
        
        for sheet_name in sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            safe_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
            csv_path = output_dir / f"{base_name}_{safe_sheet_name}.csv"
            df.to_csv(csv_path, index=False)
                
    except Exception as e:
        raise Exception(f"Error processing {excel_file.name}: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python excel_to_csv.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    logging.basicConfig(level=logging.INFO)
    convert_excel_to_csv(input_directory, output_directory)

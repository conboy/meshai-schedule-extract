#!/usr/bin/env python3
"""
Test script to convert Excel files to CSV and extract shift information using OpenRouter API.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from extract_shifts import (
    extract_shifts_from_file,
    save_json_result,
    setup_logging,
    close_session,
    get_session
)


async def convert_excel_to_csv(excel_path: str, output_dir: Optional[str] = None) -> List[str]:
    """Convert Excel file to CSV files (one per sheet)."""
    logger = logging.getLogger(__name__)
    
    excel_path = Path(excel_path)
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        return []
    
    if output_dir is None:
        output_dir = excel_path.parent / "csv_output"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Converting Excel to CSV: {excel_path}")
    
    try:
        csv_files = []
        
        # Use pandas to read all sheets
        excel_data = pd.read_excel(excel_path, sheet_name=None)
        
        for sheet_name, df in excel_data.items():
            # Clean sheet name for filename
            safe_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            csv_filename = output_dir / f"{excel_path.stem}_{safe_sheet_name}.csv"
            
            # Save to CSV
            df.to_csv(csv_filename, index=False)
            csv_files.append(str(csv_filename))
            logger.info(f"Created CSV: {csv_filename}")
        
        logger.info(f"Successfully converted {len(csv_files)} sheets to CSV")
        return csv_files
        
    except Exception as e:
        logger.error(f"Error converting Excel to CSV: {e}")
        return []


async def extract_shifts_from_csv(csv_path: str, api_key: str, model: str = "google/gemini-2.5-flash-preview-05-20") -> Optional[str]:
    """Extract shift information from CSV content using OpenRouter API."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Extracting shifts from CSV: {csv_path}")
    
    try:
        # Read CSV content
        df = pd.read_csv(csv_path)
        csv_content = df.to_string(index=False)
        
        # Prepare the prompt
        prompt = f"""Below is a CSV extracted from an Excel file using pandas. Your task is to extract ALL shifts from this CSV schedule data. Return ONLY a JSON array with these fields for each shift: Start Date, Start Time, End Date, End Time, Job, Member, Worked Hours, Note. Use null for missing fields. Be concise but complete - extract EVERY shift shown.

CSV Data:
{csv_content}"""
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "response_format": {"type": "json_object"},
            "structured_outputs": True
        }
        
        # Make request using shared session
        session = await get_session()
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        ) as response:
            
            if response.status == 200:
                result = await response.json()
                content = result['choices'][0]['message']['content']
                logger.info(f"Successfully extracted shifts from CSV")
                return content
            else:
                error_text = await response.text()
                logger.error(f"API request failed: {response.status} - {error_text}")
                return None
                
    except Exception as e:
        logger.error(f"Error extracting shifts from CSV: {e}")
        return None


async def process_excel_file(excel_path: str, api_key: str) -> bool:
    """Process Excel file: convert to CSV and extract shifts."""
    logger = logging.getLogger(__name__)
    
    try:
        # Convert Excel to CSV
        csv_files = await convert_excel_to_csv(excel_path)
        
        if not csv_files:
            logger.error("Failed to convert Excel to CSV")
            return False
        
        # Process each CSV file concurrently
        tasks = []
        for csv_file in csv_files:
            task = asyncio.create_task(extract_shifts_from_csv(csv_file, api_key))
            tasks.append((task, csv_file))
        
        success_count = 0
        for task, csv_file in tasks:
            result = await task
            
            if result:
                # Save result to JSON file
                output_file = f"{Path(csv_file).stem}_shifts.json"
                await save_json_result(result, output_file)
                logger.info(f"Saved shifts to: {output_file}")
                success_count += 1
            else:
                logger.warning(f"Failed to extract shifts from: {csv_file}")
        
        logger.info(f"Successfully processed {success_count}/{len(csv_files)} CSV files")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error processing Excel file: {e}")
        return False


async def main():
    logger = setup_logging()
    
    # Get API key from environment
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)
    
    # Find all Excel files in the data directory
    data_dir = Path("data")
    excel_files = []
    
    for pattern in ["*.xlsx", "*.xls"]:
        excel_files.extend(data_dir.glob(pattern))
    
    if not excel_files:
        logger.error("No Excel files found in data directory")
        sys.exit(1)
    
    logger.info(f"Found {len(excel_files)} Excel files to process")
    
    try:
        total_processed = 0
        total_successful = 0
        
        # Process all files concurrently
        tasks = []
        for excel_file in excel_files:
            logger.info(f"Processing: {excel_file}")
            total_processed += 1
            task = asyncio.create_task(process_excel_file(str(excel_file), api_key))
            tasks.append((task, excel_file))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*[task for task, _ in tasks], return_exceptions=True)
        
        for (task, excel_file), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process: {excel_file} - {result}")
            elif result:
                total_successful += 1
                logger.info(f"Successfully processed: {excel_file}")
            else:
                logger.error(f"Failed to process: {excel_file}")
        
        logger.info(f"Processing completed: {total_successful}/{total_processed} files successful")
        
        if total_successful == 0:
            logger.error("No files processed successfully")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up the global session
        await close_session()


if __name__ == "__main__":
    asyncio.run(main())

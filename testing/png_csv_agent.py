#!/usr/bin/env python3
"""
Testing script for png_csv_agent.py
Runs extract_shifts_from_csv_with_png on all matching CSV/PNG file pairs in testing/input_dataset/
"""

import asyncio
import os
import sys
import glob
import json
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.png_csv_agent import extract_shifts_from_csv_with_png
from session import close_session
from logger import get_logger

logger = get_logger(__name__)

async def process_single_csv_png_pair(csv_file, png_file):
    """Process a single CSV/PNG file pair."""
    filename = os.path.basename(csv_file)
    logger.info(f"Processing: {filename} with {os.path.basename(png_file)}")
    
    try:
        result = await extract_shifts_from_csv_with_png(csv_file, png_file)
        
        if result:
            return filename, {
                "status": "success",
                "csv_file": csv_file,
                "png_file": png_file,
                "data": json.loads(result) if result.strip().startswith('{') or result.strip().startswith('[') else result
            }
        else:
            logger.error(f"✗ Failed to process {filename}: No result returned")
            return filename, {
                "status": "failed",
                "csv_file": csv_file,
                "png_file": png_file,
                "error": "No result returned"
            }
            
    except Exception as e:
        logger.error(f"✗ Error processing {filename}: {e}")
        return filename, {
            "status": "error",
            "csv_file": csv_file,
            "png_file": png_file,
            "error": str(e)
        }

async def test_png_csv_agent():
    """Test png_csv_agent on all matching CSV/PNG file pairs in the input dataset."""
    logger.info("Starting PNG CSV agent testing")
    
    # Get all CSV and PNG files
    csv_dir = Path(__file__).parent / "input_dataset" / "csv"
    png_dir = Path(__file__).parent / "input_dataset" / "png"
    
    csv_files = glob.glob(str(csv_dir / "*.csv"))
    png_files = glob.glob(str(png_dir / "*.png"))
    
    if not csv_files:
        logger.error(f"No CSV files found in {csv_dir}")
        return
    
    if not png_files:
        logger.error(f"No PNG files found in {png_dir}")
        return
    
    # Create mapping of base names to file paths
    csv_map = {Path(f).stem: f for f in csv_files}
    png_map = {Path(f).stem: f for f in png_files}
    
    # Find matching pairs
    matching_pairs = []
    for csv_name, csv_path in csv_map.items():
        if csv_name in png_map:
            matching_pairs.append((csv_path, png_map[csv_name]))
        else:
            logger.warning(f"No matching PNG found for CSV: {csv_name}")
    
    if not matching_pairs:
        logger.error("No matching CSV/PNG pairs found")
        return
    
    logger.info(f"Found {len(matching_pairs)} matching CSV/PNG pairs to process")
    
    # Create tasks for all file pairs (no concurrency limit)
    tasks = [process_single_csv_png_pair(csv_file, png_file) for csv_file, png_file in sorted(matching_pairs)]
    
    # Execute all tasks concurrently
    task_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    results = {}
    successful = 0
    failed = 0
    
    for task_result in task_results:
        if isinstance(task_result, Exception):
            failed += 1
            logger.error(f"Task failed with exception: {task_result}")
        else:
            filename, result_data = task_result
            results[filename] = result_data
            if result_data["status"] == "success":
                successful += 1
                logger.info(f"✓ Successfully processed {filename}")
            else:
                failed += 1
    
    # Save results
    output_file = Path(__file__).parent / "png_csv_agent_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Testing complete: {successful} successful, {failed} failed")
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary
    print(f"\n=== PNG CSV Agent Test Summary ===")
    print(f"Total pairs: {len(matching_pairs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {output_file}")

async def main():
    """Main function to run the test."""
    try:
        await test_png_csv_agent()
    finally:
        await close_session()

if __name__ == "__main__":
    asyncio.run(main())
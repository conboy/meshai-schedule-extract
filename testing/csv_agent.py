#!/usr/bin/env python3
"""
Testing script for csv_agent.py
Runs extract_shifts_from_csv on all CSV files in testing/input_dataset/csv/
"""

import asyncio
import os
import sys
import glob
import json
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.csv_agent import extract_shifts_from_csv
from session import close_session
from logger import get_logger

logger = get_logger(__name__)

async def process_single_csv(csv_file):
    """Process a single CSV file."""
    filename = os.path.basename(csv_file)
    logger.info(f"Processing: {filename}")
    
    try:
        result = await extract_shifts_from_csv(csv_file)
        
        if result:
            return filename, {
                "status": "success",
                "data": json.loads(result) if result.strip().startswith('{') or result.strip().startswith('[') else result
            }
        else:
            logger.error(f"✗ Failed to process {filename}: No result returned")
            return filename, {
                "status": "failed",
                "error": "No result returned"
            }
            
    except Exception as e:
        logger.error(f"✗ Error processing {filename}: {e}")
        return filename, {
            "status": "error",
            "error": str(e)
        }

async def test_csv_agent():
    """Test csv_agent on all CSV files in the input dataset."""
    logger.info("Starting CSV agent testing")
    
    # Get all CSV files
    csv_dir = Path(__file__).parent / "input_dataset" / "csv"
    csv_files = glob.glob(str(csv_dir / "*.csv"))
    
    if not csv_files:
        logger.error(f"No CSV files found in {csv_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Create tasks for all files (no concurrency limit)
    tasks = [process_single_csv(csv_file) for csv_file in sorted(csv_files)]
    
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
    output_file = Path(__file__).parent / "csv_agent_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Testing complete: {successful} successful, {failed} failed")
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary
    print(f"\n=== CSV Agent Test Summary ===")
    print(f"Total files: {len(csv_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {output_file}")

async def main():
    """Main function to run the test."""
    try:
        await test_csv_agent()
    finally:
        await close_session()

if __name__ == "__main__":
    asyncio.run(main())
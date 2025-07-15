"""
Shared HTTP session management for the application
"""
import aiohttp
from logger import get_logger

logger = get_logger(__name__)

# Global session for reuse
_session = None

async def get_session():
    """Get or create the global aiohttp session."""
    global _session
    if _session is None:
        _session = aiohttp.ClientSession()
        logger.debug("Created new aiohttp session")
    return _session

async def close_session():
    """Close the global aiohttp session."""
    global _session
    if _session:
        await _session.close()
        _session = None
        logger.debug("Closed aiohttp session")
"""
Browser use tool for OpenMisty.
"""

import asyncio
from typing import Any, Dict, List, Optional

from browser_use import BrowserUse

from app.logger import logger
from app.tool import Tool


class BrowserUseTool(Tool):
    """
    Tool to browse the web.
    """
    
    name: str = "BrowserUseTool"
    description: str = "Browse the web, interact with websites, and extract information."
    
    def __init__(self, **data):
        """
        Initialize the browser use tool.
        
        Args:
            **data: Additional data to pass to the parent class
        """
        super().__init__(**data)
        self.browser_use = BrowserUse()
    
    async def run(self, url: str, task: str) -> str:
        """
        Browse the web and perform the specified task.
        
        Args:
            url: URL to browse
            task: Task to perform
            
        Returns:
            str: Result of the task
        """
        logger.info(f"Browsing {url} to perform task: {task}")
        
        try:
            # Run the browser use in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._browse, url, task
            )
            
            return result
        except Exception as e:
            logger.error(f"Error browsing the web: {str(e)}")
            return f"Error browsing the web: {str(e)}"
    
    def _browse(self, url: str, task: str) -> str:
        """
        Perform the browsing task.
        
        Args:
            url: URL to browse
            task: Task to perform
            
        Returns:
            str: Result of the task
        """
        try:
            # Perform the browsing task
            result = self.browser_use.browse(url, task)
            return result
        except Exception as e:
            logger.error(f"Error in browser use: {str(e)}")
            return f"Error in browser use: {str(e)}"
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters for the tool.
        
        Returns:
            Dict[str, Any]: Parameters for the tool
        """
        return {
            "url": {
                "type": "string",
                "description": "URL to browse",
            },
            "task": {
                "type": "string",
                "description": "Task to perform",
            },
        }
    
    def get_required_parameters(self) -> List[str]:
        """
        Get the required parameters for the tool.
        
        Returns:
            List[str]: Required parameters for the tool
        """
        return ["url", "task"] 
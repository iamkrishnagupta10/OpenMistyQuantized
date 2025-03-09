"""
File saver tool for OpenMisty.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import WORKSPACE_ROOT
from app.logger import logger
from app.tool import Tool


class FileSaver(Tool):
    """
    Tool to save files to the workspace.
    """
    
    name: str = "FileSaver"
    description: str = "Save a file to the workspace."
    
    async def run(self, filename: str, content: str) -> str:
        """
        Save a file to the workspace.
        
        Args:
            filename: Name of the file to save
            content: Content of the file
            
        Returns:
            str: Result of saving the file
        """
        logger.info(f"Saving file: {filename}")
        
        try:
            # Create the full path
            file_path = WORKSPACE_ROOT / filename
            
            # Create parent directories if they don't exist
            os.makedirs(file_path.parent, exist_ok=True)
            
            # Write the file
            with open(file_path, "w") as f:
                f.write(content)
            
            return f"File saved successfully: {filename}"
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return f"Error saving file: {str(e)}"
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters for the tool.
        
        Returns:
            Dict[str, Any]: Parameters for the tool
        """
        return {
            "filename": {
                "type": "string",
                "description": "Name of the file to save",
            },
            "content": {
                "type": "string",
                "description": "Content of the file",
            },
        }
    
    def get_required_parameters(self) -> List[str]:
        """
        Get the required parameters for the tool.
        
        Returns:
            List[str]: Required parameters for the tool
        """
        return ["filename", "content"] 
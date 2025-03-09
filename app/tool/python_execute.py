"""
Python execution tool for OpenMisty.
"""

import asyncio
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional

from app.logger import logger
from app.tool import Tool


class PythonExecute(Tool):
    """
    Tool to execute Python code.
    """
    
    name: str = "PythonExecute"
    description: str = "Execute Python code and return the result."
    
    async def run(self, code: str) -> str:
        """
        Run Python code and return the result.
        
        Args:
            code: Python code to execute
            
        Returns:
            str: Result of executing the code
        """
        logger.info(f"Executing Python code:\n{code}")
        
        # Create string buffers for stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        # Execute the code and capture stdout and stderr
        try:
            # Redirect stdout and stderr to our buffers
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Execute the code
                exec_globals = {
                    "__builtins__": __builtins__,
                    "asyncio": asyncio,
                }
                exec(code, exec_globals)
            
            # Get the output
            stdout = stdout_buffer.getvalue()
            stderr = stderr_buffer.getvalue()
            
            # Prepare the result
            result = ""
            if stdout:
                result += f"Output:\n{stdout}"
            if stderr:
                result += f"Errors:\n{stderr}"
            
            if not result:
                result = "Code executed successfully with no output."
            
            return result
        except Exception as e:
            # Get the traceback
            tb = traceback.format_exc()
            
            # Log the error
            logger.error(f"Error executing Python code: {str(e)}\n{tb}")
            
            # Return the error
            return f"Error executing Python code: {str(e)}\n{tb}"
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters for the tool.
        
        Returns:
            Dict[str, Any]: Parameters for the tool
        """
        return {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            }
        }
    
    def get_required_parameters(self) -> List[str]:
        """
        Get the required parameters for the tool.
        
        Returns:
            List[str]: Required parameters for the tool
        """
        return ["code"] 
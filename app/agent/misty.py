"""
Misty agent implementation for OpenMisty.
"""

from typing import Optional

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.llm import LLM
from app.prompt.misty import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.file_saver import FileSaver
from app.tool.google_search import GoogleSearch
from app.tool.python_execute import PythonExecute


class Misty(ToolCallAgent):
    """
    Misty is a versatile general-purpose agent that uses quantized models to solve various tasks.

    This agent extends ToolCallAgent with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    """

    name: str = "Misty"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools and quantized models"
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(), GoogleSearch(), BrowserUseTool(), FileSaver(), Terminate()
        )
    )
    
    def __init__(
        self,
        model_override: Optional[str] = None,
        quantization: Optional[str] = None,
        **data
    ):
        """
        Initialize the Misty agent.
        
        Args:
            model_override: Optional model to use instead of the one in the config
            quantization: Optional quantization level to use
            **data: Additional data to pass to the parent class
        """
        super().__init__(**data)
        
        # Override the model if specified
        if model_override or quantization:
            # Create a custom LLM instance with the specified model and quantization
            llm_config = {}
            
            if model_override:
                llm_config["model"] = model_override
                
            if quantization:
                llm_config["quantization"] = quantization
                
            # Create a new LLM instance with the custom config
            self.llm = LLM(llm_config=llm_config) 
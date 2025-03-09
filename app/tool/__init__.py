"""
Tool implementations for OpenMisty.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from app.schema import ToolDefinition


class Tool(BaseModel, ABC):
    """
    Base class for all tools.
    """
    
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of the tool")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    @abstractmethod
    async def run(self, **kwargs) -> Any:
        """
        Run the tool with the given arguments.
        
        Args:
            **kwargs: Arguments for the tool
            
        Returns:
            Any: Result of running the tool
        """
        pass
    
    def get_definition(self) -> ToolDefinition:
        """
        Get the tool definition.
        
        Returns:
            ToolDefinition: Definition of the tool
        """
        return ToolDefinition.create(
            name=self.name,
            description=self.description,
            parameters=self.get_parameters(),
            required_params=self.get_required_parameters(),
        )
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters for the tool.
        
        Returns:
            Dict[str, Any]: Parameters for the tool
        """
        pass
    
    @abstractmethod
    def get_required_parameters(self) -> List[str]:
        """
        Get the required parameters for the tool.
        
        Returns:
            List[str]: Required parameters for the tool
        """
        pass
    
    def is_termination_tool(self) -> bool:
        """
        Check if this is a termination tool.
        
        Returns:
            bool: True if this is a termination tool, False otherwise
        """
        return False


class ToolCollection:
    """
    Collection of tools.
    """
    
    def __init__(self, *tools: Tool):
        """
        Initialize the tool collection.
        
        Args:
            *tools: Tools to add to the collection
        """
        self.tools: Dict[str, Tool] = {}
        for tool in tools:
            self.add_tool(tool)
    
    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the collection.
        
        Args:
            tool: Tool to add
        """
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            Optional[Tool]: The tool, or None if not found
        """
        return self.tools.get(name)
    
    def get_tool_definitions(self) -> List[ToolDefinition]:
        """
        Get the definitions of all tools.
        
        Returns:
            List[ToolDefinition]: Definitions of all tools
        """
        return [tool.get_definition() for tool in self.tools.values()]


class Terminate(Tool):
    """
    Tool to terminate the conversation.
    """
    
    name: str = "Terminate"
    description: str = "Terminate the conversation when the task is complete."
    
    async def run(self, message: str = "Task completed.") -> str:
        """
        Run the tool.
        
        Args:
            message: Message to return
            
        Returns:
            str: The message
        """
        return message
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters for the tool.
        
        Returns:
            Dict[str, Any]: Parameters for the tool
        """
        return {
            "message": {
                "type": "string",
                "description": "Message to return",
            }
        }
    
    def get_required_parameters(self) -> List[str]:
        """
        Get the required parameters for the tool.
        
        Returns:
            List[str]: Required parameters for the tool
        """
        return []
    
    def is_termination_tool(self) -> bool:
        """
        Check if this is a termination tool.
        
        Returns:
            bool: True if this is a termination tool, False otherwise
        """
        return True 
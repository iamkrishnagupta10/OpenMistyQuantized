"""
Schema definitions for OpenMisty.
"""

from typing import Dict, List, Literal, Optional, Union, Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """
    A message in a conversation.
    """
    
    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="The role of the message sender"
    )
    content: Optional[str] = Field(None, description="The content of the message")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tool calls in the message"
    )
    tool_call_id: Optional[str] = Field(
        None, description="ID of the tool call this message is responding to"
    )
    name: Optional[str] = Field(None, description="Name of the tool")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary.
        
        Returns:
            Dict[str, Any]: The message as a dictionary
        """
        result = {"role": self.role}
        
        if self.content is not None:
            result["content"] = self.content
            
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
            
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
            
        if self.name is not None:
            result["name"] = self.name
            
        return result

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """
        Create a system message.
        
        Args:
            content: The content of the message
            
        Returns:
            Message: A system message
        """
        return cls(role="system", content=content)

    @classmethod
    def user_message(cls, content: str) -> "Message":
        """
        Create a user message.
        
        Args:
            content: The content of the message
            
        Returns:
            Message: A user message
        """
        return cls(role="user", content=content)

    @classmethod
    def assistant_message(cls, content: str) -> "Message":
        """
        Create an assistant message.
        
        Args:
            content: The content of the message
            
        Returns:
            Message: An assistant message
        """
        return cls(role="assistant", content=content)

    @classmethod
    def tool_message(
        cls, content: str, tool_call_id: str, name: Optional[str] = None
    ) -> "Message":
        """
        Create a tool message.
        
        Args:
            content: The content of the message
            tool_call_id: ID of the tool call this message is responding to
            name: Name of the tool
            
        Returns:
            Message: A tool message
        """
        return cls(
            role="tool", content=content, tool_call_id=tool_call_id, name=name
        )


class ToolParameter(BaseModel):
    """
    A parameter for a tool.
    """
    
    type: str = Field(..., description="The type of the parameter")
    description: str = Field(..., description="Description of the parameter")
    enum: Optional[List[str]] = Field(None, description="Enumeration of possible values")
    required: Optional[bool] = Field(None, description="Whether the parameter is required")


class ToolDefinition(BaseModel):
    """
    Definition of a tool.
    """
    
    type: Literal["function"] = Field("function", description="The type of the tool")
    function: Dict[str, Any] = Field(..., description="Function definition")

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required_params: Optional[List[str]] = None,
    ) -> "ToolDefinition":
        """
        Create a tool definition.
        
        Args:
            name: Name of the tool
            description: Description of the tool
            parameters: Parameters of the tool
            required_params: Required parameters
            
        Returns:
            ToolDefinition: A tool definition
        """
        function = {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
            },
        }
        
        if required_params:
            function["parameters"]["required"] = required_params
            
        return cls(type="function", function=function) 
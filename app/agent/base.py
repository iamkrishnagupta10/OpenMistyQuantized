"""
Base agent implementation for OpenMisty.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from app.llm import LLM
from app.logger import logger
from app.schema import Message


class Agent(BaseModel, ABC):
    """
    Base agent class for OpenMisty.
    
    All agents should inherit from this class and implement the run method.
    """
    
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent")
    system_prompt: str = Field(..., description="System prompt for the agent")
    llm: Optional[LLM] = Field(None, description="LLM instance")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        """Initialize the agent."""
        super().__init__(**data)
        if self.llm is None:
            self.llm = LLM()
    
    @abstractmethod
    async def run(self, prompt: str) -> Any:
        """
        Run the agent with the given prompt.
        
        Args:
            prompt: The prompt to run the agent with
            
        Returns:
            Any: The result of running the agent
        """
        pass
    
    async def _get_response(
        self,
        messages: List[Union[Dict[str, Any], Message]],
        system_msgs: Optional[List[Union[Dict[str, Any], Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Get a response from the LLM.
        
        Args:
            messages: List of messages
            system_msgs: Optional system messages
            stream: Whether to stream the response
            temperature: Sampling temperature
            
        Returns:
            str: The response from the LLM
        """
        try:
            return await self.llm.ask(
                messages=messages,
                system_msgs=system_msgs,
                stream=stream,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Error getting response from LLM: {str(e)}")
            raise 
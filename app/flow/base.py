"""
Base flow implementation for OpenMisty.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from app.agent import Agent


class FlowType(Enum):
    """
    Types of flows.
    """
    
    PLANNING = auto()
    REACT = auto()
    TOOLCALL = auto()


class Flow(BaseModel, ABC):
    """
    Base flow class for OpenMisty.
    
    All flows should inherit from this class and implement the execute method.
    """
    
    name: str = Field(..., description="Name of the flow")
    description: str = Field(..., description="Description of the flow")
    agents: Dict[str, Agent] = Field(..., description="Agents used in the flow")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    @abstractmethod
    async def execute(self, prompt: str) -> Any:
        """
        Execute the flow with the given prompt.
        
        Args:
            prompt: The prompt to execute the flow with
            
        Returns:
            Any: The result of executing the flow
        """
        pass 
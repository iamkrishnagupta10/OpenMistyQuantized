"""
Planning flow implementation for OpenMisty.
"""

import json
from typing import Any, Dict, List, Optional

from pydantic import Field

from app.agent import Agent
from app.flow.base import Flow
from app.logger import logger
from app.schema import Message


class PlanningFlow(Flow):
    """
    A flow that uses planning to solve tasks.
    """
    
    name: str = "Planning Flow"
    description: str = "A flow that uses planning to solve tasks."
    
    async def execute(self, prompt: str) -> Any:
        """
        Execute the flow with the given prompt.
        
        Args:
            prompt: The prompt to execute the flow with
            
        Returns:
            Any: The result of executing the flow
        """
        logger.info(f"Executing planning flow with prompt: {prompt}")
        
        # Get the planning agent
        agent = self.agents.get("misty")
        if agent is None:
            logger.error("Misty agent not found")
            return "Error: Misty agent not found"
        
        # Create the planning prompt
        planning_prompt = self._create_planning_prompt(prompt)
        
        # Execute the planning agent
        try:
            result = await agent.run(planning_prompt)
            return result
        except Exception as e:
            logger.error(f"Error executing planning flow: {str(e)}")
            return f"Error executing planning flow: {str(e)}"
    
    def _create_planning_prompt(self, prompt: str) -> str:
        """
        Create a planning prompt from the user prompt.
        
        Args:
            prompt: The user prompt
            
        Returns:
            str: The planning prompt
        """
        return f"""
I need to solve the following task:

{prompt}

Please help me solve this task by:
1. Breaking it down into manageable steps
2. Executing each step using the appropriate tools
3. Providing clear explanations of your reasoning and actions
4. Being honest about limitations and uncertainties
5. Focusing on delivering a practical, working solution

Let's approach this systematically and solve it step by step.
""" 
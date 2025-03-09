"""
Tool-calling agent implementation for OpenMisty.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from app.agent.base import Agent
from app.logger import logger
from app.schema import Message
from app.tool import Tool, ToolCollection


class ToolCallAgent(Agent):
    """
    An agent that can use tools to solve tasks.
    """
    
    available_tools: ToolCollection = Field(
        default_factory=ToolCollection, description="Available tools for the agent"
    )
    next_step_prompt: str = Field(
        "", description="Prompt to use for the next step in the conversation"
    )
    
    async def run(self, prompt: str) -> Any:
        """
        Run the agent with the given prompt.
        
        Args:
            prompt: The prompt to run the agent with
            
        Returns:
            Any: The result of running the agent
        """
        # Initialize conversation with system and user messages
        conversation = [
            Message.system_message(self.system_prompt),
            Message.user_message(prompt),
        ]
        
        # Get tool definitions
        tool_definitions = self.available_tools.get_tool_definitions()
        
        # Run the conversation loop
        max_iterations = 15
        for i in range(max_iterations):
            try:
                # Get response from LLM with tool calling
                response = await self.llm.ask_tool(
                    messages=conversation,
                    tools=tool_definitions,
                    tool_choice="auto",
                )
                
                # Add assistant response to conversation
                conversation.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )
                
                # Check if there are tool calls
                if not response.tool_calls:
                    # No tool calls, just return the response
                    return response.content
                
                # Process each tool call
                for tool_call in response.tool_calls:
                    # Extract tool information
                    function_call = tool_call.function
                    tool_name = function_call.name
                    tool_args_str = function_call.arguments
                    tool_call_id = tool_call.id
                    
                    # Parse tool arguments
                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        # Try to fix common JSON errors
                        fixed_args_str = self._fix_json(tool_args_str)
                        try:
                            tool_args = json.loads(fixed_args_str)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse tool arguments: {tool_args_str}")
                            tool_args = {}
                    
                    # Execute the tool
                    try:
                        tool = self.available_tools.get_tool(tool_name)
                        if tool is None:
                            raise ValueError(f"Tool not found: {tool_name}")
                        
                        logger.info(f"Executing tool: {tool_name}")
                        tool_result = await tool.run(**tool_args)
                        
                        # Add tool result to conversation
                        conversation.append(
                            Message.tool_message(
                                content=str(tool_result),
                                tool_call_id=tool_call_id,
                                name=tool_name,
                            )
                        )
                        
                        # Check if the tool is a termination tool
                        if tool.is_termination_tool():
                            return tool_result
                    except Exception as e:
                        # Add error message to conversation
                        error_message = f"Error executing tool {tool_name}: {str(e)}"
                        logger.error(error_message)
                        conversation.append(
                            Message.tool_message(
                                content=error_message,
                                tool_call_id=tool_call_id,
                                name=tool_name,
                            )
                        )
                
                # Add next step prompt if provided
                if self.next_step_prompt:
                    conversation.append(Message.user_message(self.next_step_prompt))
            except Exception as e:
                logger.error(f"Error in agent run loop: {str(e)}")
                return f"Error: {str(e)}"
        
        # Return the last assistant message if we reach the maximum iterations
        for message in reversed(conversation):
            if message.role == "assistant" and message.content:
                return message.content
        
        return "Failed to complete the task within the maximum number of iterations."
    
    def _fix_json(self, json_str: str) -> str:
        """
        Fix common JSON errors.
        
        Args:
            json_str: JSON string to fix
            
        Returns:
            str: Fixed JSON string
        """
        # Replace single quotes with double quotes
        json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
        
        # Add quotes around unquoted keys
        json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)
        
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str 
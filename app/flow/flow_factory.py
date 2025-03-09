"""
Flow factory for OpenMisty.
"""

from typing import Dict, Type

from app.agent import Agent
from app.flow.base import Flow, FlowType
from app.flow.planning import PlanningFlow
from app.logger import logger


class FlowFactory:
    """
    Factory for creating flows.
    """
    
    _flows: Dict[FlowType, Type[Flow]] = {
        FlowType.PLANNING: PlanningFlow,
    }
    
    @classmethod
    def register_flow(cls, flow_type: FlowType, flow_class: Type[Flow]) -> None:
        """
        Register a flow class for a flow type.
        
        Args:
            flow_type: Type of the flow
            flow_class: Flow class to register
        """
        cls._flows[flow_type] = flow_class
    
    @classmethod
    def create_flow(cls, flow_type: FlowType, agents: Dict[str, Agent]) -> Flow:
        """
        Create a flow of the specified type.
        
        Args:
            flow_type: Type of the flow to create
            agents: Agents to use in the flow
            
        Returns:
            Flow: The created flow
            
        Raises:
            ValueError: If the flow type is not registered
        """
        if flow_type not in cls._flows:
            logger.error(f"Flow type {flow_type} not registered")
            raise ValueError(f"Flow type {flow_type} not registered")
        
        flow_class = cls._flows[flow_type]
        return flow_class(agents=agents) 
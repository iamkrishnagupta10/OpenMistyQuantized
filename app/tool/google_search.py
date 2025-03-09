"""
Google search tool for OpenMisty.
"""

import asyncio
from typing import Any, Dict, List, Optional

from googlesearch import search

from app.logger import logger
from app.tool import Tool


class GoogleSearch(Tool):
    """
    Tool to search Google.
    """
    
    name: str = "GoogleSearch"
    description: str = "Search Google for information."
    
    async def run(self, query: str, num_results: int = 5) -> str:
        """
        Search Google for the given query.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            str: Search results
        """
        logger.info(f"Searching Google for: {query}")
        
        # Run the search in a separate thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._search, query, num_results
        )
        
        # Format the results
        if not results:
            return "No results found."
        
        formatted_results = "Search results:\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. {result}\n"
        
        return formatted_results
    
    def _search(self, query: str, num_results: int) -> List[str]:
        """
        Perform the Google search.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List[str]: Search results
        """
        try:
            # Perform the search
            results = list(search(query, num_results=num_results))
            return results
        except Exception as e:
            logger.error(f"Error searching Google: {str(e)}")
            return []
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the parameters for the tool.
        
        Returns:
            Dict[str, Any]: Parameters for the tool
        """
        return {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return",
            },
        }
    
    def get_required_parameters(self) -> List[str]:
        """
        Get the required parameters for the tool.
        
        Returns:
            List[str]: Required parameters for the tool
        """
        return ["query"] 
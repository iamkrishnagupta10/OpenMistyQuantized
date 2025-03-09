#!/usr/bin/env python3
import asyncio
import argparse
import time
import sys

from app.agent.misty import Misty
from app.flow.base import FlowType
from app.flow.flow_factory import FlowFactory
from app.logger import logger
from app.utils.hardware import check_hardware_compatibility


async def run_flow(args):
    """
    Run the planning flow with the Misty agent.
    
    Args:
        args: Command line arguments
    """
    # Check hardware compatibility and optimize for the current system
    hardware_info = check_hardware_compatibility()
    logger.info(f"Running on: {hardware_info['device_name']}")
    logger.info(f"Available memory: {hardware_info['available_memory_gb']:.2f} GB")
    
    # Initialize agents
    if args.model:
        logger.info(f"Using model: {args.model}")
        misty_agent = Misty(model_override=args.model, quantization=args.quantization)
    else:
        misty_agent = Misty()
    
    agents = {
        "misty": misty_agent,
    }
    
    # Set flow type
    flow_type = FlowType.PLANNING
    if args.flow_type:
        try:
            flow_type = FlowType[args.flow_type.upper()]
        except KeyError:
            logger.warning(f"Invalid flow type: {args.flow_type}. Using PLANNING instead.")
    
    if args.prompt:
        # Run once with the provided prompt
        logger.info(f"Processing prompt with {flow_type.name} flow: {args.prompt}")
        
        flow = FlowFactory.create_flow(
            flow_type=flow_type,
            agents=agents,
        )
        
        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                flow.execute(args.prompt),
                timeout=args.timeout,
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Request processed in {elapsed_time:.2f} seconds")
            logger.info(result)
        except asyncio.TimeoutError:
            logger.error(f"Request processing timed out after {args.timeout} seconds")
            logger.info("Operation terminated due to timeout. Please try a simpler request.")
    else:
        # Interactive mode
        logger.info(f"OpenMisty {flow_type.name} flow initialized. Enter your prompts below.")
        logger.info("Type 'exit' or press Ctrl+C to exit.")
        
        while True:
            try:
                prompt = input("\n🔮 Enter your prompt: ")
                
                if prompt.lower().strip() == "exit":
                    logger.info("Goodbye! Thank you for using OpenMisty.")
                    break
                
                if not prompt.strip():
                    logger.warning("Skipping empty prompt.")
                    continue
                
                flow = FlowFactory.create_flow(
                    flow_type=flow_type,
                    agents=agents,
                )
                
                logger.info("Processing your request...")
                
                try:
                    start_time = time.time()
                    result = await asyncio.wait_for(
                        flow.execute(prompt),
                        timeout=args.timeout,
                    )
                    elapsed_time = time.time() - start_time
                    logger.info(f"Request processed in {elapsed_time:.2f} seconds")
                    logger.info(result)
                except asyncio.TimeoutError:
                    logger.error(f"Request processing timed out after {args.timeout} seconds")
                    logger.info("Operation terminated due to timeout. Please try a simpler request.")
                
            except KeyboardInterrupt:
                logger.info("\nOperation cancelled by user.")
                break
            except Exception as e:
                logger.error(f"Error: {str(e)}")


def main_cli():
    """Command line interface entry point."""
    parser = argparse.ArgumentParser(description="OpenMisty Flow: Advanced planning and execution")
    parser.add_argument("--model", type=str, help="Specify a model to use")
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit", "none"], 
                        default="4bit", help="Quantization level")
    parser.add_argument("--flow-type", type=str, choices=["planning", "react", "toolcall"], 
                        default="planning", help="Flow type to use")
    parser.add_argument("--prompt", type=str, help="Run with a single prompt and exit")
    parser.add_argument("--timeout", type=int, default=3600, 
                        help="Timeout in seconds for the flow execution")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_flow(args))
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main_cli() 
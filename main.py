#!/usr/bin/env python3
import asyncio
import argparse
import sys

from app.agent.misty import Misty
from app.logger import logger
from app.utils.hardware import check_hardware_compatibility


async def main(args):
    """
    Main entry point for OpenMisty.
    
    Args:
        args: Command line arguments
    """
    # Check hardware compatibility and optimize for the current system
    hardware_info = check_hardware_compatibility()
    logger.info(f"Running on: {hardware_info['device_name']}")
    logger.info(f"Available memory: {hardware_info['available_memory_gb']:.2f} GB")
    
    if args.model:
        logger.info(f"Using model: {args.model}")
        agent = Misty(model_override=args.model, quantization=args.quantization)
    else:
        agent = Misty()
    
    if args.prompt:
        # Run once with the provided prompt
        logger.info(f"Processing prompt: {args.prompt}")
        await agent.run(args.prompt)
    else:
        # Interactive mode
        logger.info("OpenMisty initialized. Enter your prompts below.")
        logger.info("Type 'exit', 'quit', or press Ctrl+C to exit.")
        
        while True:
            try:
                prompt = input("\n🔮 Enter your prompt: ")
                prompt_lower = prompt.lower().strip()
                
                if prompt_lower in ["exit", "quit"]:
                    logger.info("Goodbye! Thank you for using OpenMisty.")
                    break
                    
                if not prompt.strip():
                    logger.warning("Skipping empty prompt.")
                    continue
                    
                logger.info("Processing your request...")
                await agent.run(prompt)
                
            except KeyboardInterrupt:
                logger.info("\nGoodbye! Thank you for using OpenMisty.")
                break
            except Exception as e:
                logger.error(f"Error processing prompt: {str(e)}")


def main_cli():
    """Command line interface entry point."""
    parser = argparse.ArgumentParser(description="OpenMisty: Quantized AGI for local execution")
    parser.add_argument("--model", type=str, help="Specify a model to use")
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit", "none"], 
                        default="4bit", help="Quantization level")
    parser.add_argument("--prompt", type=str, help="Run with a single prompt and exit")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main_cli() 
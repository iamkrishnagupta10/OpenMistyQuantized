"""
LLM implementation with support for both API-based and local quantized models.
"""

import os
from typing import Dict, List, Literal, Optional, Union, Any

import torch
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from app.config import LLMSettings, MODELS_ROOT, config
from app.logger import logger
from app.schema import Message


class LLM:
    """
    LLM class that supports both API-based and local quantized models.
    """
    
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "initialized"):
            llm_config = llm_config or config.llm.get(config_name, config.llm["default"])
            self.mode = llm_config.get("mode", "api")
            self.model_name = llm_config.get("model")
            self.temperature = llm_config.get("temperature", 0.0)
            
            # Initialize based on mode
            if self.mode == "local":
                self._init_local_model(llm_config)
            else:
                self._init_api_client(llm_config)
                
            self.initialized = True

    def _init_local_model(self, llm_config: Dict[str, Any]):
        """
        Initialize a local quantized model.
        
        Args:
            llm_config: LLM configuration
        """
        self.quantization = llm_config.get("quantization", "4bit")
        self.device = llm_config.get("device", "auto")
        self.max_seq_length = llm_config.get("max_seq_length", 4096)
        
        # Determine the device to use
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch, 'mps') and torch.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        logger.info(f"Initializing local model: {self.model_name}")
        logger.info(f"Quantization: {self.quantization}")
        logger.info(f"Device: {self.device}")
        
        # Create models directory if it doesn't exist
        os.makedirs(MODELS_ROOT, exist_ok=True)
        
        # Configure quantization
        if self.quantization == "4bit" and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "8bit" and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None
        
        # Load the model and tokenizer
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=MODELS_ROOT,
                trust_remote_code=True,
            )
            
            # Load model with quantization if applicable
            if quantization_config and self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=MODELS_ROOT,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=MODELS_ROOT,
                    device_map=self.device,
                    trust_remote_code=True,
                )
            
            # Create the pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=self.max_seq_length,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                device_map=self.device,
            )
            
            logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _init_api_client(self, llm_config: Dict[str, Any]):
        """
        Initialize an API client.
        
        Args:
            llm_config: LLM configuration
        """
        self.api_type = llm_config.get("api_type", "openai")
        self.api_key = llm_config.get("api_key", "")
        self.api_version = llm_config.get("api_version", "")
        self.base_url = llm_config.get("base_url", "https://api.openai.com/v1")
        self.max_tokens = llm_config.get("max_tokens", 4096)
        
        logger.info(f"Initializing API client for: {self.model_name}")
        logger.info(f"API type: {self.api_type}")
        
        if self.api_type == "azure":
            self.client = AsyncAzureOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        else:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided
        """
        formatted_messages = []

        for message in messages:
            if isinstance(message, dict):
                # If message is already a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")
                formatted_messages.append(message)
            elif isinstance(message, Message):
                # If message is a Message object, convert it to dict
                formatted_messages.append(message.to_dict())
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ["system", "user", "assistant", "tool"]:
                raise ValueError(f"Invalid role: {msg['role']}")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError(
                    "Message must contain either 'content' or 'tool_calls'"
                )

        return formatted_messages

    async def _generate_local(
        self, messages: List[dict], temperature: Optional[float] = None
    ) -> str:
        """
        Generate text using a local model.
        
        Args:
            messages: List of messages
            temperature: Sampling temperature
            
        Returns:
            str: Generated text
        """
        # Convert messages to a prompt
        prompt = self._messages_to_prompt(messages)
        
        # Generate text
        result = self.pipe(
            prompt,
            temperature=temperature or self.temperature,
            max_length=self.max_seq_length,
            do_sample=temperature > 0 if temperature is not None else self.temperature > 0,
        )
        
        # Extract the generated text
        generated_text = result[0]["generated_text"]
        
        # Remove the prompt from the generated text
        response = generated_text[len(prompt):].strip()
        
        return response

    def _messages_to_prompt(self, messages: List[dict]) -> str:
        """
        Convert messages to a prompt for local models.
        
        Args:
            messages: List of messages
            
        Returns:
            str: Prompt for the model
        """
        prompt = ""
        
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
            elif role == "tool":
                prompt += f"<|tool|>\n{content}\n"
        
        # Add the final assistant prompt
        prompt += "<|assistant|>\n"
        
        return prompt

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Format system and user messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)
            
            # Use local model if in local mode
            if self.mode == "local":
                return await self._generate_local(messages, temperature)
            
            # API mode
            if not stream:
                # Non-streaming request
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=temperature or self.temperature,
                    stream=False,
                )
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")
                return response.choices[0].message.content

            # Streaming request
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=temperature or self.temperature,
                stream=True,
            )

            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")
            return full_response

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 60,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        # Local models don't support tool calls yet, so we need to use API mode
        if self.mode == "local":
            logger.warning("Tool calls are not supported with local models. Using API mode.")
            # TODO: Implement tool calls for local models
            raise NotImplementedError("Tool calls are not supported with local models yet.")
        
        try:
            # Validate tool_choice
            if tool_choice not in ["none", "auto", "required"]:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # Set up the completion request
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
                **kwargs,
            )

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                print(response)
                raise ValueError("Invalid or empty response from LLM")

            return response.choices[0].message

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise 
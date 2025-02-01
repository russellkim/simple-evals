import os
import time
from typing import Any
from openai import OpenAI
from ..types import MessageList, SamplerBase

class PredibaseChatCompletionSampler(SamplerBase):
    """
    Sample from Predibase's chat completion API with LoRA adapter support
    """
    def __init__(
        self,
        base_url: str,
        adapter_id: str,
        api_key: str = None,
        adapter_source: str = "pbase",
        model: str = "",  # Default empty string as model name
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.api_key = api_key or os.getenv("PREDIBASE_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either through api_key parameter or PREDIBASE_API_KEY environment variable")

        if not base_url:
            raise ValueError("base_url must be provided")

        # Ensure base_url doesn't end with a slash
        self.base_url = base_url.rstrip('/')
        
        # Initialize OpenAI client with the custom base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.adapter_id = adapter_id
        self.adapter_source = adapter_source
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _pack_message(self, role: str, content: Any, is_user: bool = False):
        message = {"role": str(role), "content": content}
        if is_user:
            # Add Predibase-specific parameters for user messages
            message["parameters"] = {
                "adapter_id": self.adapter_id,
                "adapter_source": self.adapter_source
            }
        return message

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list

        messages = [
            self._pack_message(msg["role"], msg["content"], msg["role"] == "user")
            for msg in message_list
        ]

        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content

            except Exception as e:
                print(f"Error during API call: {str(e)}")
                exception_backoff = 2**trial  # exponential backoff
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
                
                # After 5 retries, give up
                if trial >= 5:
                    print("Max retries reached, returning empty string")
                    return ""

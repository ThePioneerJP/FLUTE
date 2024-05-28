# ClaudePromptProcessor.py

from typing import List, Union, Optional
from AbstractPromptProcessor import AbstractPromptProcessor

try:
    import anthropic
except ImportError:
    anthropic = None

class ClaudePromptProcessor(AbstractPromptProcessor):
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        super().__init__(api_key)
        self.model = model
        
    def generate_response(
        self,
        prompt: Union[str, List[str]],
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logprobs: Optional[int] = None,
        stream: bool = False,
        # Claude specific arguments
        metadata: Optional[dict] = None,
        stop_sequences: Optional[List[str]] = None,
        system: Optional[str] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        tools: Optional[List[dict]] = None,
        top_k: Optional[int] = None,
        user: Optional[str] = None,
    ) -> Union[str, List[str]]:
        if anthropic is None:
            raise ImportError("The 'anthropic' library is not installed. Please install it to use ClaudePromptProcessor.")
        if model is None:
            model = self.model
        
        kwargs = {
            "model": model,
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_completions": n,
            "stop_sequences": stop_sequences,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logprobs": logprobs,
            "stream": stream,
            "metadata": metadata,
            "system_prompt": system,
            "tool_choice": tool_choice,
            "tools": tools,
            "top_k": top_k,
            "user_id": user,
        }
        
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        if isinstance(prompt, str):
            prompt = [anthropic.prompt.human_prompt(remove_special_characters(prompt))]
        else:
            prompt = [anthropic.prompt.human_prompt(remove_special_characters(p)) for p in prompt]

        response = anthropic.api.generate_completions(prompt=prompt, **kwargs)
        
        if stream:
            return response
        else:
            return [choice["completion"] for choice in response["choices"]]
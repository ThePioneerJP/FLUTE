# GeminiPromptProcessor.py

from typing import List, Union, Optional
from AbstractPromptProcessor import AbstractPromptProcessor

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class GeminiPromptProcessor(AbstractPromptProcessor):
    def __init__(self, api_key: Optional[str] = None, model: str = "models/gemini-1.5-flash-latest"):
        super().__init__(api_key)
        self.model = genai.GenerativeModel(model)

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
        # Gemini specific arguments
        safety_settings: Optional[genai.SafetySettings] = None,
        generation_config: Optional[genai.GenerationConfig] = None,
        tools: Optional[genai.FunctionLibrary] = None,
        tool_config: Optional[genai.ToolConfig] = None,
        system_instruction: Optional[genai.Content] = None,
    ) -> Union[str, List[str]]:
        if genai is None:
            raise ImportError("The 'google.generativeai' library is not installed. Please install it to use GeminiPromptProcessor.")
        
        if model is not None:
            self.model = genai.GenerativeModel(model)

        contents = []
        if isinstance(prompt, str):
            contents.append(genai.Content(text=remove_special_characters(prompt)))
        else:
            contents.extend([genai.Content(text=remove_special_characters(p)) for p in prompt])

        response = self.model.generate_content(
            contents,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            safety_settings=safety_settings,
            generation_config=generation_config,
            tools=tools,
            tool_config=tool_config,
            system_instruction=system_instruction,
        )

        if stream:
            return response
        else:
            return [candidate.text for candidate in response.candidates]
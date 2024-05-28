# test_prompt_processors.py

import pytest
from dotenv import load_dotenv
import os
import sys
import pprint

sys.path.append("..")
sys.path.append("../Modules")
pprint.pprint(sys.path)

from PromptProcessorFactory import PromptProcessorFactory

load_dotenv()

def test_create_claude_prompt_processor():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    processor = PromptProcessorFactory.create_prompt_processor("claude-3-haiku-20240307", api_key=api_key)
    assert processor.api_key == api_key
    assert processor.model == "claude-3-haiku-20240307"

    response = processor.generate_response("Hello, how are you?")
    assert isinstance(response, list)
    assert len(response) > 0
    assert isinstance(response[0], str)

def test_create_gpt_prompt_processor():
    api_key = os.getenv("OPENAI_API_KEY")
    processor = PromptProcessorFactory.create_prompt_processor("gpt-4o", api_key=api_key)
    assert processor.api_key == api_key

    response = processor.generate_response("Hello, how are you?", model="gpt-4o")
    assert isinstance(response, list)
    assert len(response) > 0
    assert isinstance(response[0], str)

def test_create_gemini_prompt_processor():
    api_key = os.getenv("GOOGLE_API_KEY")
    processor = PromptProcessorFactory.create_prompt_processor("models/gemini-1.5-flash-latest", api_key=api_key)
    assert processor.api_key == api_key
    assert processor.model.model_name == "gemini-1.5-flash-latest"

    response = processor.generate_response("Hello, how are you?")
    assert isinstance(response, list)
    assert len(response) > 0
    assert isinstance(response[0], str)
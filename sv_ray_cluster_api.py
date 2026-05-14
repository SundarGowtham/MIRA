#  ------------------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------------------

import base64
import os
from functools import wraps
from urllib.parse import urlparse, unquote
import requests
from openai import OpenAI
from litellm import completion

# base_url is the base URL of the SupportVectors Ray cluster API which happens to be 10.0.10.51:8123 today.
chat_base_url = "http://10.0.10.51:8124"
chat_api_base_url = f"{chat_base_url}/v1"
_sv_openai_client = OpenAI(
        base_url=chat_api_base_url,
        api_key="sv-openai-api-key",
    )

def _is_http_url(image_source: str) -> bool:
    parsed = urlparse(image_source)
    return parsed.scheme in {"http", "https"}




def get_sv_openai_client() -> OpenAI:
    """Create an OpenAI client for the LLM served on the SupportVectors Ray cluster.
    Currently, the only supported model is "openai/gpt-oss-20b".  So, only pass this model
    when using this client for chat completions.
    Returns:
        OpenAI: An OpenAI client for the LLM (openai/gpt-oss-20b) served on the SupportVectors Ray cluster.
    """
    return _sv_openai_client

@wraps(_sv_openai_client.chat.completions.create)
def sv_openai_completion(**kwargs):
    """
    Use the SV OpenAI completion function to get a response from the LLM served on the SupportVectors Ray cluster.
    Args:
        **kwargs: Arbitrary keyword arguments
    Returns:
        dict: The response from the LLM
    """
    kwargs.pop("model", None)
    return _sv_openai_client.chat.completions.create(
        model= "openai/gpt-oss-20b",
        **kwargs,
    )

# define an sv_completion function that uses the litellm completion but hard-codes the model, api_base, and api_key
# and leaves the rest for the caller to fill in.
@wraps(completion)
def sv_completion(**kwargs):
    """ 
    Use the litellm completion function to get a response from the LLM served on the SupportVectors Ray cluster.
    Args:
        **kwargs: Arbitrary keyword arguments
    Returns:
        dict: The response from the LLM
    """
    kwargs.pop("model", None)
    kwargs.pop("api_base", None)
    kwargs.pop("api_key", None)
    return completion(
        model="openai/" + "openai/gpt-oss-20b",
        api_base=chat_api_base_url,
        api_key="sv-openai-api-key",
        **kwargs,
    )



    
def test_openai_chat():
    """Test the openai chat function"""
    response = sv_openai_completion(messages=[{"role": "user", "content": "Why is the sky blue?"}])
    print(response.choices[0].message.content)

    
if __name__ == "__main__":

    print("Testing openai chat function")
    test_openai_chat()
    print("--------------------------------")

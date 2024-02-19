from typing import List  # NOQA: UP035
import torch
#import torchvision
# Import the pipeline class from the transformers library
from transformers import pipeline
import streamlit as st
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline


# Get the huggingface token from the streamlit secrets
huggingface_token = st.secrets["huggingface_token"]
# Define the model name and some constants
MODEL_NAME = "IlyaGusev/saiga2_13b_lora"
DEFAULT_MESSAGE_TEMPLATE = "<s> {role}\n {content}</s>\n"
DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
START_TOKEN_ID = 1
BOT_TOKEN_ID = 9225

# Load the model and the tokenizer
config = PeftConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, MODEL_NAME, torch_dtype=torch.float16)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)





# Create a pipeline object with the model name and the token
model = pipeline("text-generation", model=MODEL_NAME, api_key=huggingface_token)

# Define a class to store the conversation history and generate the prompt
class Conversation:
    def __init__(
        self,
        message_template=DEFAULT_MESSAGE_TEMPLATE,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        start_token_id=START_TOKEN_ID,
        bot_token_id=BOT_TOKEN_ID,
    ):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

    def add_user_message(self, message):
        self.messages.append(
            {
                "role": "user",
                "content": message,
            }
        )

    def add_bot_message(self, message):
        self.messages.append(
            {
                "role": "bot",
                "content": message,
            }
        )

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])
        return final_text.strip()

# Define a function to create the chat completion using streamlit caching
@st.cache_data()
def create_gpt_completion(ai_model: str, messages: List[dict]) -> dict:
    # Create a conversation object with the given messages
    conversation = Conversation()
    for message in messages:
        conversation.add_user_message(message["text"])

    # Generate the prompt from the conversation history
    prompt = conversation.get_prompt(tokenizer)
    logging.info(f"{prompt=}")

    # Generate the chat completion from the prompt
    completion = model.generate(prompt, **generation_config)
    logging.info(f"{completion=}")

    # Return the completion as a dict
    return {"text": completion}


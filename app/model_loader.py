
from transformers import AutoTokenizer
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from . import settings

class ModelLoader:
    def __init__(self, base_dir, allowed_ext):
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    

    def load(self):
        # Load the pre-trained LLM model
        llm = LlamaCpp(
            model_path=settings.LLM_PATH,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            top_p=settings.LLM_TOP_P,
            callback_manager= self.callback_manager,
            verbose=True, # Verbose is required to pass to the callback manager
        )
        
        # Load the pre-trained tokenizer related to the LLM chosen and loaded in line 34
        # This tokenizer will be used convert inputs to the format the LLM expects
        # This tokenizer is used implecitly by the LLM RetrievalQA
        tokenizer = AutoTokenizer.from_pretrained(
            settings.TOKENIZATION_MODEL_PATH,
            padding=True,
            truncation=True,
            max_length=settings.TOKENIZATION_MAX_LENGTH,
            token=settings.HF_TOKEN)
        return llm
    
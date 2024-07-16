from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sentencepiece
from transformers import pipeline
# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import notebook_login
# notebook_login()

def load_model(model_name):
    if model_name == "BART":
        model_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
    elif model_name == "T5":
        model_tokenizer = AutoTokenizer.from_pretrained("t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    elif model_name == "Pegasus":
        model_tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
    elif model_name == "ProphetNet":
        model_tokenizer = AutoTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
        model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/prophetnet-large-uncased")
    elif model_name == "DistilBART":
        model_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    elif model_name == "LED":
        model_tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
        model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384")
    elif model_name == "mBART":
        model_tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-cc25")
    

    else:
        raise ValueError("Selected model is not supported.")
    
    return model, model_tokenizer

import os
os.system("python -m pip install python-multipart")

from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#To override CROS origion problem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
labels = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval",
          "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
          "relief", "remorse", "sadness", "surprise", "neutral"]
index_to_labels = {index: label for index, label in enumerate(labels)}

#Install required libraries
@app.get("/install_dependencies/")
def install_dependencies_api():
	try:
		os.system("apt-get update")
		os.system("apt-get upgrade")
		os.system("python -m pip install --upgrade pip")
		os.system("python -m pip install transformers==4.3.2")
		os.system("python -m pip install folium==0.2.1")
		os.system("python -m pip install imgaug==0.2.6")
		os.system("python -m pip install numpy==1.19.2")
		os.system("python -m pip install --no-cache-dir --default-timeout=100 --upgrade torch")
		return {"message": "Installed!"}
	except Exception as e:
		return {"message" : str(e)}


@app.get("/")
def read_root():
    return {"message": "NLP is up and listning"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

	
@app.get("/load_model_binary/")
def load_model_binary_api():
	from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
	global loaded_model_binary
	global tokenizer_binary	
	config = AutoConfig.from_pretrained(model_name, num_labels=2)
	model_tmp = AutoModelForSequenceClassification.from_config(config=config)
	loaded_model_binary = model_tmp.from_pretrained("nlp_model_binary.pt")
	tokenizer_binary = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	return {"message": "Binary model loaded"}
	 

@app.get("/detect_emotion_binary/")
def detect_emotion_binary_api(message: str):
	try:
		input = tokenizer_binary(message, return_tensors="pt")
		output = loaded_model_binary(**input)
		p = output[0].cpu().data.numpy().argmax()
		res = "Positive" if p == 0 else "Negative"
		return {"message":res}
	except Exception as e:
		return {"message" : str(e)}
		
		
@app.get("/load_model_full/")
def load_model_full_api():
	from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
	global loaded_model_full
	global tokenizer_full
	config = AutoConfig.from_pretrained(model_name, num_labels=27)
	model_tmp = AutoModelForSequenceClassification.from_config(config=config)
	loaded_model_full = model_tmp.from_pretrained("nlp_model_full.pt")
	tokenizer_full = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	return {"message": "Full model loaded"}
	 
		
		
@app.get("/detect_emotion_full/")
def detect_emotion_full_api(message: str):
	try:
		input = tokenizer_full(message, return_tensors="pt")
		output = loaded_model_full(**input)
		p = output[0].cpu().data.numpy().argmax()
		res = index_to_labels[p]
		return {"message":res}
	except Exception as e:
		return {"message" : str(e)}		
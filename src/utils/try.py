import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm   
import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
import pandas as pd
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import groq
from dotenv import load_dotenv
import json
import os
import csv
import torch
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    AutoConfig,
    pipeline 
)
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import groq
from dotenv import load_dotenv
import os
import gc
import streamlit as st
import torch
from huggingface_hub import login
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
    pipeline
)
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import groq
from dotenv import load_dotenv
import os
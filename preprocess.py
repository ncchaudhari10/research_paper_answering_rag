from pymilvus import model
from docling.datamodel.pipeline_options import PdfPipelineOptions

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

import warnings
warnings.filterwarnings("ignore")


import os
from dotenv import load_dotenv
import json

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
import pandas as pd
from openai import OpenAI

class Preprocess:

    def __init__(self,file_path=None):
        

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options.use_gpu = True  # <-- set this.
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            })

        self.text_splitter = RecursiveCharacterTextSplitter(

            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
            )

        self.file_path=file_path
        if file_path:
            self.data = self.setup()
        load_dotenv()
        OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

        self.openai = OpenAI()
        self.model="text-embedding-3-small"
        self.openai_ef = model.dense.OpenAIEmbeddingFunction(
            model_name='text-embedding-3-small', # Specify the model name
            api_key=OPENAI_API_KEY, # Provide your OpenAI API key
            dimensions=1536 # Set the embedding dimensionality
        )

    def setup(self):

        files = glob.glob(self.file_path)
        df = pd.DataFrame(columns=['title','meta','text'])

        for file_path in files:
            print("performing :",file_path)

            result = self.converter.convert(file_path)

            document_content = result.document.export_to_markdown()

            parsed_dict= self.parse_markdown_to_dict(document_content)

            temp = list(parsed_dict.keys())
            paper_title = temp[0] if (len(temp)>0) else ""

            df_temp= self.perform_chunking(parsed_dict)
            title_data =[paper_title]*(len(df_temp))
            df_temp['title'] = title_data

            df = pd.concat([df,df_temp])

        df['title'] = df['title'].astype(str)
        df['meta'] = df['meta'].astype(str)
        df['text'] = df['text'].astype(str)
        
        return df


    def parse_markdown_to_dict(self,markdown_text):
        hierarchy = {}
        current_key = None
        current_value = []

        for line in markdown_text.splitlines(keepends=True):
            # Check if the line is a header
            header_match = re.match(r'^(#{1,6})\s+(.*)', line)
            if header_match:
                if current_key:
                    # Save the previous section
                    hierarchy[current_key] = ''.join(current_value).strip()
                current_key = header_match.group(2).strip()
                current_value = []  # Reset for new section
            elif current_key:
                current_value.append(line)  # Append line as-is to the current section

        # Save the last section
        if current_key:
            hierarchy[current_key] = ''.join(current_value).strip()

        return hierarchy

    
    def perform_chunking(self,data):

        text_data=[]
        meta_data=[]


        for key,val in data.items():
            if(key=='References'):
                continue

            texts = self.text_splitter.split_text(val)

            for i in range(len(texts)):
                texts[i] = key + ": " + texts[i]
            
                meta_data.append(key)
                text_data.append(texts[i])
            
        
        return pd.DataFrame({'meta': meta_data,
            'text': text_data
            })

    def get_embedding(self,docs):
        return self.openai_ef.encode_documents(docs)


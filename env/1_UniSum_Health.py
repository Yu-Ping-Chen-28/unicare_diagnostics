import os
import boto3
import sagemaker
import streamlit as st
import logging
from streamlit.components.v1 import html
from langchain.llms.bedrock import Bedrock
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain



st.set_page_config(
    page_title="UniSum Health",
    page_icon=":🦄",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .block-container {padding-top: 1rem;padding-bottom: 0rem;padding-left: 5rem;padding-right: 5rem}
    h1 {text-align: center;}
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


target_region = os.environ.get("AWS_REGION")    
    
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=target_region,
    endpoint_url='https://bedrock-runtime.us-west-2.amazonaws.com'
)

s3 = boto3.client(
    service_name='s3',
    region_name=target_region,
)

textract_client = boto3.client("textract", region_name=target_region)

modelId = "anthropic.claude-instant-v1"


### Complete this section, TODO fill in values
# Details here https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=anthropic.claude-instant-v1
llm = Bedrock(
    model_id=modelId,
    model_kwargs={
        "max_tokens_to_sample":300,
        "temperature":0.8,
        "top_k":50,
        "top_p":0.8,
        "stop_sequences":[]
    },
    client=bedrock_runtime,
)


def read_file_from_s3(bucket_name, file_name):
    obj = s3.get_object(Bucket=bucket_name, Key=file_name)
    data = obj['Body'].read()
    return data


def summarize_text(data):  
    
    ### Learn about LangChain MapReduce Here https://python.langchain.com/docs/use_cases/summarization#option-2-map-reduce
    ### Complete this section, fill out the TODO with the text you want to summarize

    map_prompt = """Write a concise medical transcript summary of the following:
    {text}
    CONCISE SUMMARY:"""
    
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    
    combine_prompt = """
    Write a concise summary of of the following text.
    Return your response in bullet points which covers the reason for visit, history of present illness, assessment, and plan of the text. Anonymize sensitive information.
    {text}
    BULLET POINT SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=False
    )
    
    
    output = summary_chain.run(data)     
    st.write(output.strip())  
    st.divider() 



if __name__ == "__main__":

    st.title(":rainbow[UniCare Generative AI]")

    st.header(":stethoscope: :blue[UniSum Health]")
    st.subheader("_Your Medical Summary Assistant_", divider='rainbow')
    st.write("Our clinicians are busy! Please help us summarize medical transcripts so that we can focus on improving patient experience.")
    
    sess = sagemaker.Session()
    s3_bucket = sess.default_bucket()
    
    uploaded_file = st.file_uploader(label="Upload a transcript PDF here.")
    
    if uploaded_file is not None:
        s3.upload_fileobj(uploaded_file, s3_bucket, uploaded_file.name)
        file_path = f's3://{s3_bucket}/{uploaded_file.name}'
        st.write(f'Successfully uploaded to {file_path}')

        
        with st.spinner(text="Working on it..."):
            loader = AmazonTextractPDFLoader(file_path, client=textract_client)
            documents = loader.load_and_split()
            #print(documents)
            summarize_text(documents)
            
    


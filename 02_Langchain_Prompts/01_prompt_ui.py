# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt
import streamlit as st

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

load_dotenv()
# model = ChatOpenAI()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 300,
        "do_sample": True
    }
)

model = ChatHuggingFace(llm=llm)

st.header('Research Tool ')

paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"]) 

style_input = st.selectbox("Select Explanation Style" ,["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Type", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

# template

# template = PromptTemplate(
#     template="""
# Please Summarize the reasearch paper titled "{paper_input}" with the following specifications :
# Explanation Style : {style_input}
# Explanation Length: {length_input}
# 1.Mathematical Details: 
#     - Include relevant mathematical equations if present in the paper.
#     - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
# 2.Analogies:
#     -Use relatable analogies to simplify complex ideas.
# If cetain information is not available in the paper, respond with "Insufficient Information available" intead of guessing
# """,
# input_variables=['paper_input','style_input','length_input'],
# validate_template=True
# )

template = load_prompt('template.json')

#filling the place holders 
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input':style_input,
    'length_input': length_input
})

if st.button("Summarize"):
    with st.spinner("Generating summary ..."):
        prompt_value = template.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
    })


        prompt_text = prompt_value.to_string()

        result = model.invoke(prompt_text)
        st.write(result.content)

# if st.button('Summarize'):
#     chain = template | model
#     result = chain.invoke({
#         'paper_input':paper_input,
#         'style_input':style_input,
#         'length_input':length_input
#     })
#     st.write(result.content)
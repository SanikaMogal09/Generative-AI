from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
Please Summarize the reasearch paper titled "{paper_input}" with the following specifications :
Explanation Style : {style_input}
Explanation Length: {length_input}
1.Mathematical Details: 
    - Include relevant mathematical equations if present in the paper.
    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2.Analogies:
    -Use relatable analogies to simplify complex ideas.
If cetain information is not available in the paper, respond with "Insufficient Information available" intead of guessing
""",
input_variables=['paper_input','style_input','length_input'],
validate_template=True
)

template.save('template.json')
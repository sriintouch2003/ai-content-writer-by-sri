import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import logging

# Configure logging for debugging
logging.basicConfig(
    filename="app.log",
    encoding="utf-8",
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

## Function To get response from LLAma 2 model


def getLLamaresponse(input_text, no_words, blog_style):
    ### LLama2 model
    llm = CTransformers(
        model="models/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        config={"max_new_tokens": 256, "temperature": 0.01},
    )

    ## Prompt Template

    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
            """

    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"], template=template
    )

    ## Generate the ressponse from the LLama 2 model
    response = llm(
        prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
    )
    logger.info(response)
    return response


# 4. Use streamlit to create a web app
def main():
    from PIL import Image

    # Loading Image using PIL
    ai_icon = Image.open("ai.png")

    st.set_page_config(
        page_title="Adapto Blogs By Sri",
        page_icon=ai_icon,
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    hide_streamlit_style = """

    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>

    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.markdown(
        "<h2 style='text-align: center;'><i>AdaptoBlogs</i></h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h4 style='text-align: center;'><i>Tailored AI-Generated Content for Targetted Audience</i></h4><br>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h6 style='text-align: center;'><i>Please grace us with the central theme you envision for our next compelling blog narrative by entering text below.</i></h6>",
        unsafe_allow_html=True,
    )
    input_text = st.text_input("")
    ## creating to more columns for additonal 2 fields

    col1, col2 = st.columns([5, 5])

    with col1:
        no_words = st.text_input("Number of Words")
    with col2:
        blog_style = st.selectbox(
            "Writing the blog content for",
            ("Common People", "Researchers", "Data Scientist"),
            index=0,
        )

    submit = st.button("Generate Blog Content")

    ## Final response
    if submit:
        st.write(getLLamaresponse(input_text, no_words, blog_style))


if __name__ == "__main__":
    main()

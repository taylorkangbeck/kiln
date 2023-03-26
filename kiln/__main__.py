import json
import os
import webbrowser

import click
import requests
import yaml
from dotenv import load_dotenv
from langchain.agents import initialize_agent, load_tools
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# from scrapeghost import SchemaScraper

load_dotenv() # loads openai api key from .env file

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
.card {{
  box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
  transition: 0.3s;
  width: 50%;
  border-radius: 5px;
}}

.card:hover {{
  box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
}}

img {{
  border-radius: 5px 5px 0 0;
}}

.container {{
  padding: 2px 16px;
}}
</style>
</head>
<body>

<div class="card">
  <img src="{img_src}" style="width:100%">
  <div class="container">
    <p>{styled_transcript}</p> 
  </div>
</div>

</body>
</html> 
"""

@click.command()
def main():
    # Scrape URLs for core content
    # with open("urls.txt", "r") as file:
    #     urls = file.readlines()
    # scrape_article = SchemaScraper(
    #     schema={
    #         "url": "url",
    #         "headline": "string",
    #         "summary": "string",
    #         "content": "string",
    #     }
    # )
    with open("content.yaml", "r") as file:
        content = yaml.safe_load(file)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.9)
    summarize_prompt = PromptTemplate(
        input_variables=["source_body"],
        template="Summarize the following post into a headline which is less than 10 words: {source_body}",
    )
    summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)
    reference_prompt = PromptTemplate(
        input_variables=["post_transcript"],
        template="You are researchGPT an world class research assistant. You receive a body of text. You will analyze each sentence of the body of text. You need to determine if the sentence is referring to a reference not contained in the body of text. Return the same body of text, but with html underline tags (<u> and </u>) wrapping the reference. Body of text: {post_transcript}",
    )
    reference_chain = LLMChain(llm=llm, prompt=reference_prompt) 
    style_prompt = PromptTemplate(
        input_variables=["post_transcript"],
        template="You receive a body of text. Return the same body of text, but with some changes for words that require emphasis. Use uppercase for exclamations and reactions. Use html bold tags (<b> and </b>) for emotions. Use html italics tags (<i> and </i>) for words that the speaker is stressing. At the end of sentences, also insert relevant emojis that represent either the emotion of the sentence or something referenced in the sentence. Body of text: {post_transcript}",
    )
    style_chain = LLMChain(llm=llm, prompt=style_prompt)
    image_gen_prompting_prompt = PromptTemplate(
        input_variables=["post_transcript"],
        template="You receive the transcript of a conversational message. Pick one object from the conversation. Write a six word or less description of an image of the object. It should match the theme of the transcript. The description should be a good prompt for an image generation AI model. Do not include numbers or dates in the response. Transcript: {post_transcript}")
    image_gen_prompting_chain = LLMChain(llm=llm, prompt=image_gen_prompting_prompt)

    # for source in content['sources']:
    #     source_summary = summarize_chain.run(source['body'])
    #     # TODO store in vector db
    
    for index, post in enumerate(content['posts']):
        # post_summary = summarize_chain.run(post['transcript'])
        # print(post_summary)
        # post['summary'] = post_summary
      img_prompt = image_gen_prompting_chain.run(post['transcript'])
      img_src = get_image(img_prompt)

      underlined_transcript = reference_chain.run(post['transcript'])
  #     # TODO look up in vector db and store references
      styled_transcript = style_chain.run(underlined_transcript)
    
      html = HTML_TEMPLATE.format(
          img_src=img_src,
          styled_transcript=styled_transcript
      )
      html_filename = f'card_{index}.html'
      with open(html_filename, "w") as file:
          file.write(html)
          print(html_filename)
      webbrowser.open(html_filename, new=0, autoraise=True)



    


    # tools = load_tools(["requests"], llm=llm)
    # agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    # for url in urls:
    #     agent(f'Extract the primary text content from this html code and summarize it: {url}')


    
def get_image(prompt):
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    data = {
        "prompt": prompt + ", minimal line art style, no text or numbers",
        "n": 1,
        "size": "256x256",
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # If you want to view the response as JSON
    response_json = response.json()
    print(prompt)
    url = response_json['data'][0]['url']
    return url
        



if __name__ == '__main__':
    main()
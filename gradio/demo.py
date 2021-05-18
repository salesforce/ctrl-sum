from summarizers import Summarizers
import gradio as gr

summ = Summarizers('normal')

def ctrlsum(text):
  contents = text
  return summ(contents)

inputs = gr.inputs.Textbox(lines=5, label="Input Text")
outputs =  gr.outputs.Textbox(label="CTRLsum")

title = "CTRLsum"
description = "demo for Salesforce CTRLsum. To use it, simply input text or click one of the examples text to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2012.04281'>CTRLsum: Towards Generic Controllable Text Summarization</a> | <a href='https://github.com/salesforce/ctrl-sum'>Github Repo</a></p>"
examples = [
            ["My name is Kevin. I love dogs. I loved dogs from 1996. Today, I'm going to walk on street with my dogs"]
]

gr.Interface(ctrlsum, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()
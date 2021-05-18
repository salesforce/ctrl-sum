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
            ["""Tunip is the Octonauts' head cook and gardener. 
He is a Vegimal, a half-animal, half-vegetable creature capable of breathing on land as well as underwater. 
Tunip is very childish and innocent, always wanting to help the Octonauts in any way he can. 
He is the smallest main character in the Octonauts crew."""]
]

gr.Interface(ctrlsum, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()
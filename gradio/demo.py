from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast
import gradio as gr

model = AutoModelForSeq2SeqLM.from_pretrained("hyunwoongko/ctrlsum-cnndm")

tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/ctrlsum-cnndm")


def ctrlsum(text):
  data = tokenizer(text, return_tensors="pt")
  input_ids, attention_mask = data["input_ids"], data["attention_mask"]
  return tokenizer.batch_decode(model.generate(input_ids, attention_mask=attention_mask, num_beams=5))[0].replace("</s>","")

inputs = gr.inputs.Textbox(lines=5, label="Input Text")
outputs =  gr.outputs.Textbox(label="CTRLsum")

title = "CTRLsum"
description = "demo for Salesforce CTRLsum. To use it, simply input text or click one of the examples text to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2012.04281'>CTRLsum: Towards Generic Controllable Text Summarization</a> | <a href='https://github.com/salesforce/ctrl-sum'>Github Repo</a></p>"
examples = [
            ["My name is Kevin. I love dogs. I loved dogs from 1996. Today, I'm going to walk on street with my dogs"]
]

gr.Interface(ctrlsum, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()
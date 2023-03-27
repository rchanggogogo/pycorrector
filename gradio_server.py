# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/3/23 5:21 PM
==================================="""

# -*- coding: utf-8 -*-

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gradio as gr
import operator
import torch
from transformers import BertTokenizerFast, BertForMaskedLM
from gradio.components import HighlightedText, JSON
from examples.corrector_pipeline import correct_text


def ai_text(text):
    # with torch.no_grad():zhe
    #     outputs = model(**tokenizer([text], padding=True, return_tensors='pt'))
    corrected_text, details = correct_text(text)
    def to_highlight(corrected_sent, errs):
        if errs:
            output = [{"entity": "纠错", "word": err[1], "start": err[2], "end": err[3]} for i, err in
                    enumerate(errs)]
        else:
            output = [{"entity": "无错", "word": corrected_sent, "start": 0, "end": len(corrected_sent)}]
        return {"text": corrected_sent, "entities": output}

    def get_errors(corrected_text, origin_text):
        sub_details = []
        for i, ori_char in enumerate(origin_text):
            if ori_char in [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤']:
                # add unk word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                continue
            if i >= len(corrected_text):
                continue
            if ori_char != corrected_text[i]:
                if ori_char.lower() == corrected_text[i]:
                    # pass english upper char
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                    continue
                sub_details.append((ori_char, corrected_text[i], i, i + 1))
        sub_details = sorted(sub_details, key=operator.itemgetter(2))
        return corrected_text, sub_details

    # _text = tokenizer.decode(torch.argmax(outputs.logits[0], dim=-1), skip_special_tokens=True).replace(' ', '')
    # corrected_text = _text[:len(text)]

    if details:
        highlight = [{'entity': '纠错', 'word': text[err[1]: err[2]], 'start': err[1], 'end': err[2]+1} for i, err in enumerate(details)]
    else:
        highlight = [{'entity': '纠错', 'word': corrected_text, 'start': 0, 'end': 0}]
    print(text, ' => ', corrected_text, details)
    # outputs = to_highlight(corrected_text, details)

    return {"text": corrected_text, "entities": highlight}, details


if __name__ == '__main__':
    # print(ai_text('少先队员因该为老人让坐'))

    examples = [
        ['真麻烦你了。希望你们好好的跳无'],
        ['少先队员因该为老人让坐'],
        ['机七学习是人工智能领遇最能体现智能的一个分支'],
        ['今天心情很好'],
        ['他法语说的很好，的语也不错'],
        ['他们的吵翻很不错，再说他们做的咖喱鸡也好吃'],
    ]

    gr.Interface(
        ai_text,
        inputs="textbox",
        outputs=[
            HighlightedText(
                label="Output",
                show_legend=True,
            ),
            JSON(
                label="JSON Output"
            )
        ],
        title="Chinese Spelling Correction",
        description="Copy or input error Chinese text. Submit and the machine will correct text.",
        article="Link to <a href='https://github.com/shibing624/pycorrector' style='color:blue;' target='_blank\'>Github REPO</a>",
        examples=examples
    ).launch(share=True, server_name='0.0.0.0', server_port=8081)

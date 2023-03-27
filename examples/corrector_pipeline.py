# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/3/13 4:00 PM
==================================="""
import os
import time

import torch

from pycorrector import Corrector
from pycorrector.detector import ErrorType, Detector
import operator
from pycorrector.utils.text_utils import get_errors, get_editops
from pycorrector.t5.t5_corrector import T5Corrector
from pycorrector.macbert.macbert_corrector import MacBertCorrector
from pycorrector.bart.bart_corrector import BartCorrector
from modelscope.pipelines import pipeline
from modelscope.hub.snapshot_download import snapshot_download
from pycorrector.utils.tokenizer import segment, split_2_short_text
from loguru import logger
from pycorrector import config

# model_dir = snapshot_download('damo/nlp_bart_text-error-correction_chinese', cache_dir='/8t/workspace/lchang/models', revision='v1.0.1')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

lm_corrector = Corrector(custom_confusion_path_or_dict=config.user_custom_confusion_path)
# t5_corrector = T5Corrector(device_=device)
macbert_corrector = MacBertCorrector(device_=device)

bart_corrector = BartCorrector(model_dir='/8t/workspace/lchang/models/damo/nlp_bart_text-error-correction_chinese', device=device)


# async time cost
# lm time cost:  3.652263879776001
# t5 time cost:  0.9497480392456055
# macbert time cost:  0.01278376579284668
# time cost:  4.615200519561768
# async def lm_corrector_func(text):
#     start_time = time.time()
#     corrected_sent, detail = lm_corrector.correct(text)
#     end_time = time.time()
#     print('lm time cost: ', end_time - start_time)
#     return corrected_sent, detail
#
#
# async def t5_corrector_func(text):
#     start_time = time.time()
#     corrected_sent, detail = t5_corrector.t5_correct(text)
#     end_time = time.time()
#     print('t5 time cost: ', end_time - start_time)
#
#     return corrected_sent, detail
#
#
# async def macbert_corrector_func(text):
#     start_time = time.time()
#     corrected_sent, detail = macbert_corrector.macbert_correct(text)
#     end_time = time.time()
#     print('macbert time cost: ', end_time - start_time)
#     return corrected_sent, detail
#
#
# async def correct_text(text):
#     start_time = time.time()
#     results = await asyncio.gather(
#         lm_corrector_func(text),
#         t5_corrector_func(text),
#         macbert_corrector_func(text)
#     )
#     end_time = time.time()
#     print('time cost: ', (end_time - start_time))
#
#     for result in results:
#         print(result)


def lm_corrector_func(text):
    corrected_sent, detail = lm_corrector.correct(text)
    logger.debug(f"lm result: {corrected_sent}, {detail}")
    return corrected_sent, detail, 'lm'


# def t5_corrector_func(text):
#     corrected_sent, detail = t5_corrector.t5_correct(text)
#     logger.debug(f"t5 result: {corrected_sent}, {detail}")
#     return corrected_sent, detail, 't5'


def macbert_corrector_func(text):
    corrected_sent, detail = macbert_corrector.macbert_correct(text)
    logger.debug(f"macbert result: {corrected_sent}, {detail}")
    return corrected_sent, detail, 'macbert'


def bart_correct(text):
    corrected_sent, sub_details = bart_corrector.bart_correct(text, maxlen=128)
    logger.debug(f"bart result: {corrected_sent}, {sub_details}")
    return corrected_sent, sub_details, 'bart'


def correct_text(text, weight=0.9):
    # 先进行定制纠错
    sentence, _ = lm_corrector.custom_corrector(text)
    # 再进行拼写纠错
    sentence, _, _ = macbert_corrector_func(sentence)
    logger.debug(f"spelling correction result: {sentence}")
    # 最后进行语法纠错
    sentence, _, _ = bart_correct(sentence)
    logger.debug(f"grammatical correction result: {sentence}")

    origin_ppl_score = lm_corrector.ppl_score(segment(text, cut_type='char'))
    logger.debug(f"origin ppl score: {origin_ppl_score}")
    corrected_ppl_score = lm_corrector.ppl_score(segment(sentence, cut_type='char'))
    logger.debug(f"corrected ppl score: {corrected_ppl_score}")

    # 如果纠错后的句子的ppl值小于原句的ppl值的weight倍，则认为纠错成功
    if corrected_ppl_score < origin_ppl_score * weight:
        return get_editops(text, sentence)
    else:
        return text, []



# def glm_correct(text):
#     start_time = time.time()
#     result = glm_corrector(text)
#     end_time = time.time()
#     print('glm time cost: ', end_time - start_time)
#     print(f"glm result: {result}")
#     return result['output'], []

def save_result(model_name='lm', raw_srs=None, raw_tgt=None, result=None):
    with open(f'{model_name}_result.txt', 'w') as f:
        for srs, tgt, res in zip(raw_srs, raw_tgt, result):
            f.write(f"{srs}\t{tgt}\t{res}\n")


if __name__ == '__main__':
    text = "他法语说的很好，德语也不错"
    # text = "我前几天在上网看报纸的时候看过一个关于一个明星的善行的内容，她是一名歌手，她以前也帮助过孤儿当了他们的姐姐，这次她又决定帮助非洲的一个女孩子。因为战争父母都失去了以后那个女孩儿来养她的弟弟听起来觉得很难帮她似的但是你有心意帮她们就很容易，一个月送她一点的钱，有时间写信等等那些钱也是一个月只是两百块左右的。"
    # print(correct_text(text))
    de = Detector()
    sentence = segment(text, cut_type='word')
    sentence_char = segment(text, cut_type='char')
    print(sentence)
    print(sentence_char)
    print(f"word ppl score: {de.ppl_score(sentence)}")
    print(f"char ppl score: {de.ppl_score(sentence_char)}")
    print(f"text ppl score: {de.lm.perplexity(text)}")
    print(f"word ngram score: {de.ngram_score(sentence)}")
    print(f"char ngram score: {de.ngram_score(sentence_char)}")
    print(f"text ngram score: {de.lm.score(text, bos=False, eos=False)}")
    scores = de.lm.full_scores(text)

    for score in scores:
        print(score)
    # print(de.lm.full_scores(text))
    # with open('GEC_test_500.txt', 'r') as f:
    #
    #     for i, text in enumerate(f):
    #         splits = text.split('\t')
    #         if i > 10:
    #             break
    #
    #         text = splits[1]
    #         print(text)
    #         correct_text(text)
    #         print('########################')
    # else:
    #     for t in range(0, len(splits), 2):
    #         text = splits[t]
    #         print(text)
    #         correct_text(text)
    #         print('########################')

    # asyncio.run(correct_text(text))
    # correct_text(text)

import sys
import os
import argparse
import logging
import configparser
import time
from html.parser import HTMLParser
import Levenshtein
import nltk
from rank_bm25 import BM25Okapi
import numpy as np


def parse_output(output):
    result = []
    items = output.split(' | ')
    for item in items:
        if not item:
            continue
        fields = item.split('#')

        if len(fields) != 4:
            continue
        prev_sentence = find_between(fields[0], 'Previous Sentence: ', None)
        q_type = find_between(fields[1], 'Type: ', None)
        answer = find_between(fields[2], 'Answer Keywords: ', None)
        question = find_between(fields[3], 'Question: ', None)

        if not all([prev_sentence, q_type, answer, question]):
            continue
        result.append({
            'position': None,
            'answer': answer,
            'type': q_type,
            'question': question,
            'prev_sentence': prev_sentence
        })

    return result


def map_back(article, pred_sentences):
    if type(article) != list:
        article_sentences = nltk.sent_tokenize(article)
    else:
        article_sentences = article

    candidates = [nltk.word_tokenize(sentence) for sentence in article_sentences]
    bm25 = BM25Okapi(candidates)

    pre_ids = []
    for pred_sent in pred_sentences:
        tokenized_query = nltk.word_tokenize(pred_sent)
        candidate_scores = bm25.get_scores(tokenized_query)
        sorted_idx = np.argsort(candidate_scores)
        pre_ids.append(sorted_idx[-1])

    return article_sentences, pre_ids


def merge_list(lists):
    result = []
    for lst in lists:
        result.extend(lst)
    return result


def ascii_encode(x):
    # str => int list
    result = []
    for char in x:
        result.append(ord(char))
    return result


def ascii_decode(x):
    # int list => str
    result = []
    for number in x:
        result.append(chr(number))
    return ''.join(result)


def find_between(string, prefix, suffix, default=''):
    if prefix is None:
        start = 0
    else:
        start = string.find(prefix)

    if suffix is None:
        end = len(string)
    else:
        end = string.find(suffix)
    # print(string, prefix, start, suffix, end)
    if start == -1 or end == -1:
        return default

    return string[start+len(prefix): end]


def n_grams(list_a, list_b, n, target):
    assert target in ['overlap', 'novel']
    list_a_ngrams = {}
    list_b_ngrams = {}

    total_n_gram = len(list_b) - n + 1
    hit_n_gram = 0
    for i in range(0, len(list_a), n):
        n_gram = '#'.join(list_a[i:i + n])
        if n_gram not in list_a_ngrams:
            list_a_ngrams[n_gram] = 0
        list_a_ngrams[n_gram] += 1
    for j in range(0, len(list_b), n):
        n_gram = '#'.join(list_b[j:j + n])
        if n_gram not in list_b_ngrams:
            list_b_ngrams[n_gram] = 0
        list_b_ngrams[n_gram] += 1
    for n_gram in list_b_ngrams:
        if n_gram in list_a_ngrams:
            hit_n_gram += list_b_ngrams[n_gram]
    if target == 'overlap':
        return hit_n_gram / total_n_gram
    else:
        return 1 - (hit_n_gram / total_n_gram)


def f_beta_score(p, r, beta=1):
    return (1 + pow(beta, 2)) * p * r / ((pow(beta, 2)*p + r) + 1e-10)


def to_percentage(number, reserved=3):
    return '{}{}'.format(round(number, reserved)*100, '%')


def info_line(info, symbol='=', total_length=200):
    prefix_len = (total_length - len(info) - 2) // 2
    suffix_len = (total_length - len(info) - 2) // 2

    if prefix_len + suffix_len + len(info) + 2 < total_length:
        suffix_len += 1

    return '{} {} {}'.format(symbol*prefix_len, info, symbol*suffix_len)


def to_date(timestamp):
    time_array = time.localtime(int(timestamp))
    time_format = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    return time_format


def lev_ratio(target, candidate):
    return Levenshtein.ratio(target, candidate)

def auto_convert(x):
    try:
        float_x = float(x)
    except Exception:
        float_x = None

    try:
        int_x = int(x)
    except Exception:
        int_x = None

    if int_x is not None and int_x == float_x:
        return int_x
    elif float_x is not None and int_x != float_x:
        return float_x
    else:
        return x


def is_int(x):
    try:
        int(x)
        return True
    except Exception:
        return False


def format_message(messages):
    fmt_msgs = []
    for msg in messages:
        fmt_msgs.append('{}: {}'.format(msg['role'], msg['content']))
    return fmt_msgs


class ArticleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = None
        self.tag_stack = []
        self.cur_attr = None  # name, value
        self.paragraphs = []
        self.paragraphs_plain = []

    def handle_starttag(self, tag, attrs) -> None:
        # print('Now, handle start tag {}'.format(tag))
        self.tag_stack.append(tag)
        if self.tag_stack[-1] == 'p':
            self.paragraphs.append({
                'segments': [],
                'num_question': 0,
            })
        if attrs:
            # print('here', attrs[0][0], attrs[0][1], type(attrs[0][1]))
            self.cur_attr = attrs[0]

        if self.tag_stack[-1] == 'span' and self.cur_attr[0] == 'id' and self.cur_attr[1].startswith('question_'):
            question_id = int(self.cur_attr[1].split('_')[-1])
            self.paragraphs[-1]['segments'].append(['question', question_id])
            self.paragraphs[-1]['num_question'] += 1

    def handle_endtag(self, tag: str) -> None:
        assert tag == self.tag_stack[-1]
        self.tag_stack.pop(-1)
        self.cur_attr = None

    def handle_data(self, data: str) -> None:
        # print('here', self.tag_stack[-1], data,)
        if len(self.tag_stack) == 0:
            return
        if self.tag_stack[-1] == 'p':
            self.paragraphs[-1]['segments'].append(['plain_text', data])
            # print('paragraph: {} text: {}'.format(len(self.paragraphs), data))
        elif self.tag_stack[-1] == 'span':
            if self.cur_attr and self.cur_attr == ('class', 'peng_answer'):
                self.paragraphs[-1]['segments'].append(['answer', data])
                # print('paragraph: {} span_answer: {}'.format(len(self.paragraphs), data))
            # elif self.cur_attr and self.cur_attr[0] == 'id' and self.cur_attr[1].startswith('question_'):
            #     print('There, found question')
            #     question_id = int(self.cur_attr[1].split('_')[-1])
            #     self.paragraphs[-1]['segments'].append(['question', question_id])
            #     self.paragraphs[-1]['segments']['num_question'] += 1
            #     # print('paragraph: {} span_question: {}'.format(len(self.paragraphs), question_id))
        elif self.tag_stack[-1] == 'title':
            self.title = data

    def set_plain_paragraphs(self):
        for para in self.paragraphs:
            text = ''
            for seg_type, seg_text in para['segments']:
                if seg_type == 'plain_text':
                    text += seg_text
            self.paragraphs_plain.append(text)

if __name__ == '__main__':
    test_page = """
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <h1>Hello, World!</h1>
        <p>This is a <a href="http://example.com">link</a> in a paragraph. <span id="124">This </span></p>
        <p class="description">This is another paragraph with a class attribute.</p>
    </body>
    </html>
    """
    with open('data/evaluation/authentic/articles/classical_chinese_philosophy.html', 'r') as fp:
        test_page = '\n'.join(fp.readlines())
    parser = ArticleParser()
    parser.feed(test_page)
# def setup():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--conf', type=str, default='config/euler_conf.ini')
#     parser.add_argument('--job', type=str, default='eval_qg')
#     parser.add_argument('--d', type=bool, default=True)
#     parser.add_argument('--h', type=bool, default=False)
#     parser.add_argument('--s', type=bool, default=True)
#     parser.add_argument('--a', type=bool, default=True)
#     parser.add_argument('--d_type', type=str, default='continuous')  #
#     parser.add_argument('--d_source', type=str, default='kt')  # kt, gd
#     parser.add_argument('--decoding', type=str, default='normal')  # normal, constrained
#     parser.add_argument('--joint', type=bool, default=False)
#     parser.add_argument('--inc', type=bool, default=False)
#     parser.add_argument('--temperature', type=float, default=2.0)
#
#     args, remaining_argv = parser.parse_known_args()
#     config = configparser.ConfigParser()
#     config.read(args.conf)
#
#     for section in config:
#         if section == 'DEFAULT':
#             continue
#         for option in config.options(section):
#             value = auto_convert(config.get(section, option))
#             parser.add_argument('--{}'.format(option), default=value, type=type(value))
#
#     args = parser.parse_args(remaining_argv)
#
#     logging.basicConfig(
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         # filename='train_kt.log',  # 'kt_log_reg/train_kt_0.5_0.log',  # '{}.log'.format(args.job),  # args.kt_train_log,
#         level=logging.INFO,
#         filemode='w'
#     )
#     return args


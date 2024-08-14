import copy
import math
import pprint
import random
import sys
import json
import os
from common_utils import map_back, info_line, ArticleParser, find_between, n_grams, parse_output
from nltk import word_tokenize
import logging
import torch
from openai import OpenAI
import nltk
from tqdm import tqdm
import evaluate
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_bm25 import BM25Okapi
import numpy as np
from config import local, euler
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm
from evaluate import load

bertscore = load("bertscore")

def remove_dup(output_items):
    question_set = set([])
    new_items = []
    for item in output_items:
        question = item['question'].strip()
        if question in question_set:
            continue
        question_set.add(question)
        item['question'] = question
        new_items.append(copy.deepcopy(item))
    return new_items


class CrossEvaluator:
    def __init__(self, output_file, reference_file, rouge=True, bleu=True, meteor=True, b_score=False, num=True):
        self.data = {}

        self.cal_rouge = rouge
        self.cal_bleu = bleu
        self.cal_meteor = meteor
        self.cal_b_score = b_score

        # format: {id: 'article_id', 'article': 'article', 'article_sentences': [], 'output': [{'position:' 'sid', 'type': 'q_type', 'prev_sentence': prev_sentence, 'question': question}]}
        with open(output_file, 'r') as fp:
            for line in fp.readlines():
                data = json.loads(line.strip())
                # items = output['generated'].split('[SEP]')
                # generated_questions = []
                # for idx, item in enumerate(items):
                #     if idx != len(item) - 1:
                #         question = find_between(item, prefix='Question: ', suffix='[SEP]').strip()
                #     else:
                #         question = find_between(item, prefix='Question: ', suffix='[SEP]').strip()
                #     generated_questions.append(question)
                # if output['id'] not in self.data:
                #     self.data[output['id']] = {}
                if data['id'] not in self.data:
                    self.data[data['id']] = {}
                # print('there', data)
                data['output'] = remove_dup(data['output'])
                self.data[data['id']]['output'] = ' '.join([item['question'] for item in data['output']]).strip()
                if not self.data[data['id']]['output']:
                    self.data[data['id']]['output'] = 'No Questions'

        with open(reference_file, 'r') as fp:
            for line in fp.readlines():
                data = json.loads(line.strip())
                # items = data['output'].split('[SEP]')
                # reference_questions = []
                # for item in items:
                #     if idx != len(item) - 1:
                #         question = find_between(item, prefix='Question: ', suffix='[SEP]').strip()
                #     else:
                #         question = find_between(item, prefix='Question: ', suffix='[SEP]').strip()
                #     reference_questions.append(question)
                if data['id'] not in self.data:
                    continue
                self.data[data['id']]['reference'] = ' '.join([item['question'] for item in data['output']])

        logging.info('Collect {} examples for evaluation'.format(len(self.data)))
        self.result = {
            'Rouge-L': [],
            'BLEU': [],
            'METEOR': [],
            'BertScore': [],
        }

        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load("bleu")
        self.meteor = evaluate.load('meteor')
        self.bert_score = evaluate.load("bertscore")
        self.new_rouge = Rouge()

    def cross_eval(self):
        # Rouge-L, BLEU, METEOR, BERT-SCORE
        for key, data in tqdm(self.data.items()):
            if self.cal_rouge:
                rouge_scores = self.rouge.compute(references=[data['reference']], predictions=[data['output']])
                # print('here', rouge_scores, self.result['Rouge-L'])
                self.result['Rouge-L'].append(rouge_scores['rougeL'])
            if self.cal_bleu:
                bleu_scores = self.bleu.compute(references=[[data['reference']]], predictions=[data['output']])
                self.result['BLEU'].append(bleu_scores['bleu'])

            if self.cal_meteor:
                meteor_scores = self.meteor.compute(references=[data['reference']], predictions=[data['output']])
                self.result['METEOR'].append(meteor_scores['meteor'])
            if self.cal_b_score:
                bert_scores = self.bert_score.compute(references=[data['reference']], predictions=[data['output']], lang='en')
                self.result['BertScore'].append(bert_scores['f1'][0])
        # print('corpus_bleu', corpus_bleu_score)

        for metric in self.result:
            self.result[metric] = sum(self.result[metric]) / (len(self.result[metric]) + 1e-10)
            # print(metric, self.result[metric])
            # print('-'*200)

        if self.cal_bleu:
            corpus_bleu_score = corpus_bleu(
                [[data['reference']] for data in self.data.values()],
                [data['output'] for data in self.data.values()],
                weights=[0.25, 0.25, 0.25, 0.25])
            self.result['corpus_bleu'] = corpus_bleu_score


class SummaryEvaluator:
    def __init__(self, article_files, coherence_prompt_file, informativeness_prompt_file, consistency_prompt_file, summary_file, temperature, prompt_model, user_blacklist, score_output, max_request_num=3):
        with open(coherence_prompt_file, 'r') as fp:
            self.coherence_eval_prompt = ''.join(fp.readlines())
        with open(informativeness_prompt_file, 'r') as fp:
            self.informativeness_eval_prompt = ''.join(fp.readlines())
        with open(consistency_prompt_file, 'r') as fp:
            self.consistency_eval_prompt = ''.join(fp.readlines())
        self.temperature = temperature
        self.prompt_model = prompt_model
        self.client = OpenAI()
        self.max_request_num = max_request_num
        self.score_output = score_output

        self.qa_prompt = '''Read the article and answer the given question. Your answer should solely be based on the article without introducing any external knowledge. Use the exact words from the article whenever possible. Get straight to the answer without any preamble or digression. Do not repeat the question.
Article: {article}
Question: {question}
Your answer:'''

        self.articles = {}
        for article_file in article_files:
            article_path = os.path.join('data/evaluation/control/articles', '{}.html'.format(article_file))
            with open(article_path, 'r') as fp:
                article_title = os.path.split(article_file)[-1].replace('.html', '')
                article_page = '\n'.join(fp.readlines())
                article_parser = ArticleParser()
                article_parser.feed(article_page)
                article_parser.set_plain_paragraphs()
                self.articles[article_title] = article_parser
                # print('='*100)
                # print('\n'.join(article_parser.paragraphs_plain))
                # exit(1)

        logging.info('Reading {} articles: {}'.format(len(self.articles), self.articles.keys()))

        self.questions = {group_id: {} for group_id in ['authentic', 'generated']}  # {'group': {'title': []}}
        name_map = {'authentic': 'authentic', 'generated': 'prompt_gpt'}
        for group_id in self.questions:
            for title in self.articles:
                # article_text = '\n'.join(self.articles[title].paragraphs_plain)
                self.questions[group_id][title] = []
                with open('data/evaluation/{}/guiding_questions/{}.json'.format(name_map[group_id], title), 'r') as fp:
                    questions = json.load(fp)
                    for question in questions['questions_intext']:
                        item = {'question': question['question']}
                        self.questions[group_id][title].append(item)
                print('='*100)
                print('Group {}, Article {}, Questions:'.format(group_id, title))
                for question in self.questions[group_id][title]:
                    print(question)

        self.answers = {'authentic': {title: {} for title in self.articles}, 'generated': {title: {} for title in self.articles}}  # group, title, question
        for group_id in self.answers:
            for title in self.answers[group_id]:
                answer_file = 'data/evaluation/{}/answers/{}.jsonl'.format(name_map[group_id], title)
                if os.path.exists(answer_file):
                    print('loading answers from {}'.format(answer_file))
                    with open(answer_file, 'r') as fp:
                        answers = json.loads(fp.readlines()[0])
                        for question, answer in answers.items():
                            self.answers[group_id][title][question] = answer
                else:
                    article = '\n'.join(self.articles[title].paragraphs_plain)
                    questions = [item['question'] for item in self.questions[group_id][title]]
                    answers = self.__get_answer_from_llm(article=article, questions=questions)
                    with open(answer_file, 'w') as fp:
                        fp.write(json.dumps({
                            questions[idx]: answers[idx] for idx in range(len(questions))
                        }))
        #         exit(1)
        # exit(1)
        self.group_summaries = {group_id: {title: {} for title in self.articles} for group_id in ['control', 'authentic', 'generated']}  # group name -> article name -> user_prolific_id
        with open(summary_file, 'r') as fp:
            for line in fp.readlines():
                data = json.loads(line.strip())
                if 'prolific_pid' not in data or data['prolific_pid'] in user_blacklist:
                    continue
                if data['phase'] != 3:
                    continue
                summary = data['text_0']
                summary_words = nltk.word_tokenize(summary)
                user_group = data['uid'].split('_')[-1]
                title = data['aid'].split('#')[0]
                if len(self.group_summaries[user_group][title]) >= 3:
                    continue
                self.group_summaries[user_group][title][data['prolific_pid']] = {
                    'text': summary,
                    'words': summary_words,
                    'overlap': {'uni_gram': 0, 'bi_gram': 0, 'tri_gram': 0},
                    'length': len(summary_words),
                    'quality': {metric: {'rationale': None, 'score': 0} for metric in ['coherence', 'consistency', 'informativeness']}
                }
        num_control_summaries = sum([len(values) for values in self.group_summaries['control'].values()])
        num_authentic_summaries = sum([len(values) for values in self.group_summaries['authentic'].values()])
        num_generated_summaries = sum([len(values) for values in self.group_summaries['generated'].values()])
        logging.info('Collect {} summaries in total, control {}, generated {}, authentic {}'.format(
            num_control_summaries + num_authentic_summaries + num_generated_summaries,
            num_control_summaries,
            num_generated_summaries,
            num_authentic_summaries
        ))

        # for group in self.group_summaries:
        #     for title in self.group_summaries[group]:
        #         print('Number:', group, title, len(self.group_summaries[group][title]))
        #
        # exit(1)
        # for group in self.group_summaries:
        #     for title in self.group_summaries[group]:
        #         for pid in self.group_summaries[group][title]:
        #             if 'text' not in self.group_summaries[group][title][pid]:
        #                 print(group, title, pid, self.group_summaries[group][title][pid])

        # self.eval_overlap()
        # self.eval_quality()
        # pprint.pprint(self.group_summaries)

        # self.get_average()
        # exit(1)
        self.eval_sq_rel_new()

    def eval_quality(self, ):
        # LLM as evaluator: coherence, consistency, informativeness.
        # ngram overlap w/ article, ngram w/ question-relevant
        score_fp = open(self.score_output, 'w')
        for article_title in tqdm(self.articles):
            article = self.articles[article_title]
            article_text = ' '.join(article.paragraphs_plain)
            for batch_id in range(3):
                batch_inputs = []
                for user_group in ['control', 'generated', 'authentic']:
                    items = list(self.group_summaries[user_group][article_title].items())  #  (user_id, summary)
                    # print('here', items)
                    batch_inputs.append([user_group, items[batch_id][0], items[batch_id][1]['text']])  # (group_id, user_id, summary_text)

                random.shuffle(batch_inputs)  # shuffle group
                batch_group_ids, batch_user_ids, batch_summaries = list(zip(*batch_inputs))
                # print('group_ids', batch_group_ids)
                # print('user_ids', batch_user_ids)
                # print('summaries', batch_summaries)

                # coherence
                coherence_rationales, coherence_scores = self.__lm_eval(
                    prompt=self.coherence_eval_prompt,
                    title=article.title,
                    article=article_text,
                    batch_summaries=batch_summaries,
                    metric='Coherence'
                )

                for idx in range(3):
                    self.group_summaries[batch_group_ids[idx]][article_title][batch_user_ids[idx]]['quality']['coherence'] = {
                        'rationale': coherence_rationales[idx],
                        'score': coherence_scores[idx]
                    }
                # consistency
                consistency_rationale, consistency_scores = self.__lm_eval(
                    prompt=self.consistency_eval_prompt,
                    title=article.title,
                    article=article_text,
                    batch_summaries=batch_summaries,
                    metric='Consistency'
                )
                for idx in range(3):
                    self.group_summaries[batch_group_ids[idx]][article_title][batch_user_ids[idx]]['quality']['consistency'] = {
                        'rationale': consistency_rationale[idx],
                        'score': consistency_scores[idx]
                    }

                # informativeness
                informativeness_rationales, informativeness_scores = self.__lm_eval(
                    prompt=self.informativeness_eval_prompt,
                    title=article.title,
                    article=article_text,
                    batch_summaries=batch_summaries,
                    metric='Informativeness'
                )
                for idx in range(3):
                    self.group_summaries[batch_group_ids[idx]][article_title][batch_user_ids[idx]]['quality']['informativeness'] = {
                        'rationale': informativeness_rationales[idx],
                        'score': informativeness_scores[idx]
                    }
                for idx in range(3):
                    score_fp.write(json.dumps({
                        'id': '{}#{}#{}'.format(batch_group_ids[idx], article_title, batch_user_ids[idx]),
                        'summary': batch_summaries[idx],
                        'informativeness': {'score': informativeness_scores[idx], 'rationale': informativeness_rationales[idx]},
                        'coherence': {'score': coherence_scores[idx], 'rationale': coherence_rationales[idx]},
                        'consistency': {'score': consistency_scores[idx], 'rationale': consistency_rationale[idx]}
                    })+'\n')

        score_fp.close()

    def __lm_eval(self, prompt, metric, title, article, batch_summaries):
        messages = [{'role': 'user', 'content': prompt.format(title=title, article=article, summary_1=batch_summaries[0], summary_2=batch_summaries[1], summary_3=batch_summaries[2])}]
        # print(info_line('{} Prompt'.format(metric)))
        # print(messages[0]['content'])
        # print('prompt', messages[0]['content'])
        response = self.client.chat.completions.create(
            model=self.prompt_model,
            messages=messages,
            temperature=self.temperature
        )
        response_content = response.choices[0].message.content
        # print(info_line('{} Response'.format(metric)))
        # print(response_content)
        rationale_1 = find_between(response_content, 'Analysis of summary 1: ', 'Analysis of summary 2')
        rationale_2 = find_between(response_content, 'Analysis of summary 2: ', 'Analysis of summary 3')
        rationale_3 = find_between(response_content, 'Analysis of summary 3: ', '{} Scores:'.format(metric))

        score_1 = find_between(response_content, 'Score for summary 1:', 'Score for summary 2').strip()
        score_2 = find_between(response_content, 'Score for summary 2:', 'Score for summary 3').strip()
        score_3 = find_between(response_content, 'Score for summary 3:', '## The end of Output ##').strip()

        rationales = [rationale_1, rationale_2, rationale_3]
        scores = [float(score_1), float(score_2), float(score_3)]

        # print('rationales', rationales)
        # print('scores', scores)
        # exit(1)
        return rationales, scores

    def eval_overlap(self):
        for group_id in self.group_summaries:
            for article_title in self.group_summaries[group_id]:
                for pid in self.group_summaries[group_id][article_title]:
                    article_text = ' '.join(self.articles[article_title].paragraphs_plain)
                    # print(self.articles[article_title].paragraphs_plain)
                    # print(self.articles[article_title])
                    # print('here', article_text)
                    # exit(1)
                    summary = self.group_summaries[group_id][article_title][pid]['text']

                    summary_words = nltk.word_tokenize(summary)
                    article_words = nltk.word_tokenize(article_text)

                    overlap_uni_gram = n_grams(article_words, summary_words, n=1, target='overlap')
                    overlap_bi_gram = n_grams(article_words, summary_words, n=2, target='overlap')
                    overlap_tri_gram = n_grams(article_words, summary_words, n=3, target='overlap')
                    self.group_summaries[group_id][article_title][pid]['overlap'] = {
                        'uni_gram': overlap_uni_gram,
                        'bi_gram': overlap_bi_gram,
                        'tri_gram': overlap_tri_gram
                    }

    def get_average(self):
        average = {group: {
            'quality': {'coherence': [], 'consistency': [], 'informativeness': []},
            'overlap': {'uni_gram': [], 'bi_gram': [], 'tri_gram': []}
        } for group in ['control', 'authentic', 'generated']}
        for group_id in self.group_summaries:
            for article_title in self.group_summaries[group_id]:
                for pid in self.group_summaries[group_id][article_title]:
                    average[group_id]['quality']['coherence'].append(self.group_summaries[group_id][article_title][pid]['quality']['coherence']['score'])
                    average[group_id]['quality']['informativeness'].append(self.group_summaries[group_id][article_title][pid]['quality']['informativeness']['score'])
                    average[group_id]['quality']['consistency'].append(self.group_summaries[group_id][article_title][pid]['quality']['consistency']['score'])
                    average[group_id]['overlap']['uni_gram'].append(self.group_summaries[group_id][article_title][pid]['overlap']['uni_gram'])
                    average[group_id]['overlap']['bi_gram'].append(self.group_summaries[group_id][article_title][pid]['overlap']['bi_gram'])
                    average[group_id]['overlap']['tri_gram'].append(self.group_summaries[group_id][article_title][pid]['overlap']['tri_gram'])

        for group in average:
            for key in average[group]['quality']:
                # print(average[group]['quality'][key])
                # exit(1)
                average[group]['quality'][key] = sum(average[group]['quality'][key]) / len(average[group]['quality'][key])
            for key in average[group]['overlap']:
                average[group]['overlap'][key] = sum(average[group]['overlap'][key]) / len(average[group]['overlap'][key])

        pprint.pprint(average)

    def eval_sq_rel(self):
        # summary-question relationship
        all_answers = {}
        print('retrieving answer sentences')
        for group in self.questions:
            for title in self.questions[group]:
                article = '\n'.join(self.articles[title].paragraphs_plain)
                article_sentences = nltk.sent_tokenize(article)
                questions = [item['question'] for item in self.questions[group][title]]

                answers = []
                for question in questions:
                    answers.append(self.answers[group][title][question])
                answer_sent_ids = []
                # print('{} Extract Answer {}'.format('='*50, '='*50))
                for idx, question in enumerate(questions):
                    # print('#'*100)
                    # print('Question: {}\n Answer: {}'.format(question, answers[idx]))
                    sent_ids = retrieve_sentences(article_sentences, answers[idx])
                    answer_sent_ids.append(sent_ids)
                    # print('Selected sentences: {}'.format([article_sentences[i] for i in sent_ids]))
                # exit(1)
                answer_concat = []
                for ids in answer_sent_ids:
                    answer_concat.extend(ids)
                answer_concat = list(set(answer_concat))
                answer_concat.sort()
                answer_sentences = [article_sentences[i] for i in answer_concat]
                answer_sentences = ' '.join(answer_sentences)
                a_id = '{}${}'.format(title, group)
                all_answers[a_id] = answer_sentences

        all_summaries = {}
        for group in self.group_summaries:
            for title in self.group_summaries[group]:
                for user_id in self.group_summaries[group][title]:
                    s_id = '{}${}${}'.format(title, group, user_id)
                    all_summaries[s_id] = self.group_summaries[group][title][user_id]

        scores = {}
        print('Calculating BertScore ...')
        rouge = Rouge()
        for a_id in all_answers:
            for s_id in all_summaries:
                a_title, a_group = a_id.split('$')
                s_title, s_group, s_user_id = s_id.split('$')
                if a_title != s_title:
                    continue
                summary = all_summaries[s_id]['text']
                answer_llm = all_answers[a_id]
                # print('Summary', summary['text'])
                # print('Answer LLm', answer_llm)
                # match_score = bertscore.compute(predictions=[summary], references=[answer_llm], model_type='bert-base-uncased', lang='en')['recall'][0]
                match_score = rouge.get_scores(summary, answer_llm)[0]
                match_score = (match_score['rouge-1']['p'] + match_score['rouge-2']['p'] + match_score['rouge-l']['p']) /3
                # print('bertscore', bert_score_recall)
                # exit(1)
                cross_id = '{}${}'.format(s_group, a_group)
                if cross_id not in scores:
                    scores[cross_id] = []
                scores[cross_id].append(match_score)

        print('cross_scores', scores)

        for cross_id in scores:
            scores[cross_id] = sum(scores[cross_id]) / len(scores[cross_id])

        for cross_id in scores:
            print(cross_id, scores[cross_id])

    def eval_sq_rel_new(self):
        all_summaries = {}
        for group in self.group_summaries:
            for title in self.group_summaries[group]:
                for user_id in self.group_summaries[group][title]:
                    s_id = '{}${}${}'.format(title, group, user_id)
                    all_summaries[s_id] = self.group_summaries[group][title][user_id]

        all_answers = {}
        for group in self.questions:
            for title in self.questions[group]:
                # article = '\n'.join(self.articles[title].paragraphs_plain)
                questions = [item['question'] for item in self.questions[group][title]]
                for question in questions:
                    ans_id = '{}${}${}'.format(title, group, question)
                    all_answers[ans_id] = self.answers[group][title][question]

        pairs = [
            ('authentic', 'authentic'),
            ('authentic', 'generated'),
            ('generated', 'generated'),
            ('generated', 'authentic'),
            ('control', 'authentic'),
            ('control', 'generated')
        ]

        result = {}
        # pair, title, user_id: [q1, ..., q_n]
        print('calculating scores')
        for target_sum_group, target_ans_group in pairs:
            result[(target_sum_group, target_ans_group)] = {}
            for sum_id in all_summaries:
                sum_title, sum_group, sum_uid = sum_id.split('$')
                # print(sum_group, target_sum_group)
                if sum_group != target_sum_group:
                    continue
                for ans_id in all_answers:
                    ans_title, ans_group, question = ans_id.split('$')
                    if ans_group != target_ans_group:
                        continue
                    if sum_title != ans_title:
                        continue
                    if sum_title not in result[(target_sum_group, target_ans_group)]:
                        result[(target_sum_group, target_ans_group)][sum_title] = {}
                    if sum_uid not in result[(target_sum_group, target_ans_group)][sum_title]:
                        result[(target_sum_group, target_ans_group)][sum_title][sum_uid] = []
                    score = bertscore.compute(predictions=[all_summaries[sum_id]['text']], references=[all_answers[ans_id]], model_type='bert-base-uncased', lang='en')['recall'][0]
                    result[(target_sum_group, target_ans_group)][sum_title][sum_uid].append(score)
        # print(result)
        # exit(1)
        avg_score = {}
        for sum_group, ans_group in result:
            if (sum_group, ans_group) not in avg_score:
                avg_score[(sum_group, ans_group)] = []
            for title in result[(sum_group, ans_group)]:
                for uid in result[(sum_group, ans_group)][title]:
                    avg_score[(sum_group, ans_group)].append(sum(result[(sum_group, ans_group)][title][uid])/len(result[(sum_group, ans_group)][title][uid]))

            avg_score[(sum_group, ans_group)] = sum(avg_score[(sum_group, ans_group)]) / len(avg_score[(sum_group, ans_group)])
        print('avg_score', avg_score)

        max_score = {}
        for sum_group, ans_group in result:
            if (sum_group, ans_group) not in max_score:
                max_score[(sum_group, ans_group)] = []
            for title in result[(sum_group, ans_group)]:
                for uid in result[(sum_group, ans_group)][title]:
                    max_score[(sum_group, ans_group)].append(max(result[(sum_group, ans_group)][title][uid]))

            max_score[(sum_group, ans_group)] = sum(max_score[(sum_group, ans_group)]) / len(max_score[(sum_group, ans_group)])
        print('max_score', max_score)

        # top2_score = {}
        # print()

    def __get_answer_from_tag(self):
        pass

    def __get_answer_from_llm(self, questions, article):
        results = []
        for question in questions:
            messages = [{'role': 'user', 'content': self.qa_prompt.format(article=article, question=question)}]
            # print(info_line('{} Prompt'.format(metric)))
            # print(messages[0]['content'])
            # print('prompt', messages[0]['content'])
            # print(messages)
            # exit(1)
            response = self.client.chat.completions.create(
                model=self.prompt_model,
                messages=messages,
                temperature=self.temperature
            )
            answer = response.choices[0].message.content
            results.append(answer)
            print('~'*200)
            print(question, answer)
            # exit(1)

        return results


def retrieve_sentences(article_sentences, answer):
    rouge = Rouge()
    selected_sentences = []
    cur_rouge = 0
    while True:
        best_new_id = None
        for s_id, sentence in enumerate(article_sentences):
            if s_id in selected_sentences:
                continue
            selected_sentences.append(s_id)
            selected_sentences.sort()
            cur_text = ' '.join([article_sentences[i] for i in selected_sentences])
            new_rouge = rouge.get_scores(cur_text, answer)[0]
            new_rouge = (new_rouge['rouge-1']['f'] + new_rouge['rouge-2']['f'] + new_rouge['rouge-l']['f']) / 3
            if new_rouge > cur_rouge:
                best_new_id = s_id
                cur_rouge = new_rouge
            selected_sentences.remove(s_id)
        if best_new_id:
            selected_sentences.append(best_new_id)
            selected_sentences.sort()
        else:
            break
    selected_sentences.sort()
    return selected_sentences


def format_evaluation_data(article_title, output_file, user_group, model_dir, post_test='summarization', base_dir='data/evaluation'):
    assert user_group in ['control', 'generated', 'authentic']

    result = {
        'id': '{}#{}'.format(article_title, user_group),
        'user_group': user_group
    }

    with open(os.path.join(base_dir, model_dir, 'articles', article_title + '.html'), 'r') as fp:
        article = ''.join(fp.readlines())

    if user_group == 'control':
        result.update({'questions_intext': []})
    else:
        with open(os.path.join(base_dir, model_dir, 'guiding_questions', article_title+'.json'), 'r') as fp:
            guiding_questions = json.load(fp)
            result.update(guiding_questions)

    result['article'] = article
    if post_test == 'questions':
        with open(os.path.join(base_dir, 'evaluation_questions', article_title+'.json'), 'r') as fp:
            evaluation_questions = json.load(fp)
    else:
        evaluation_questions = {'questions_performance': [
            {'question': "Please provide a summary according to your best effort. It should be based solely on what you remember from reading the text and no external sources. Your authentic response is crucial to our study. Minimum 100 words (currently <span id='summary_word_count'>0</span>).", 'answer': '', 'is_asked': 'N/A'}
        ]}
    result.update(evaluation_questions)

    with open(output_file+'_{}.jsonl'.format(user_group), 'w') as fp:
        json.dump(result, fp)

    # article_fp = open(os.path.join(base_path, 'article_with_questions_grouthtruth', article_title+'.html'), 'r')
    # question_fp = open(os.path.join(base_path, 'groundtruth_questions', article_title+'.json'), 'r')
    #
    # article = ''.join(article_fp.readlines())
    # questions = json.load(question_fp)
    # questions['article'] = article
    #
    # with open(output_file, 'w') as fp:
    #     json.dump(questions, fp)

#
# class Perplexity:
#     def __init__(self, model_name):
#         self.model = AutoModelForCausalLM.from_pretrained(model_name)
#         self.model.eval()
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#     def calculate(self, input_file, key):
#         acc_ppl_question = 0
#         acc_question_tokens = 0
#
#         acc_ppl_next_sentence = 0
#         acc_next_sentence_tokens = 0
#
#         with open(input_file, 'r') as fp:
#             for line in tqdm(fp.readlines()):
#                 data = json.loads(line.strip())
#                 questions = data[key]
#
#                 for question in questions:
#                     if not all(question.values()):
#                         continue
#                     print(question)
#                     question_id = question['question_id']
#                     question_text = question['question']
#                     pre_sentence = None
#                     if question_id != 0:
#                         pre_sentence = data['article'][question_id-1]
#                     next_sentence = None
#                     if question_id != len(data['article']):
#                         next_sentence = data['article'][question_id]
#
#                     if pre_sentence:
#                         q_ppl, q_tokens = self.__call_model(pre_sentence, question_text)
#                         acc_ppl_question += q_ppl
#                         acc_question_tokens += q_tokens
#                     if next_sentence:
#                         n_ppl, n_tokens = self.__call_model(pre_sentence, next_sentence)
#                         acc_ppl_next_sentence += n_ppl
#                         acc_next_sentence_tokens += n_tokens
#
#         print({
#             'question perplexity': acc_ppl_question / acc_question_tokens,
#             'next sentence perplexity': acc_ppl_next_sentence / acc_next_sentence_tokens
#         })
#
#     def __call_model(self, sentence_1, sentence_2):
#         encoded_1 = self.tokenizer(sentence_1, return_tensors='pt')
#         encoded_2 = self.tokenizer(sentence_2, return_tensors='pt')
#         input_ids = torch.cat([encoded_1['input_ids'], encoded_2['input_ids']], -1)
#         attention_mask = torch.cat([encoded_1['attention_mask'], encoded_2['attention_mask']], -1)
#         # print(attention_mask)
#         labels = torch.tensor([[-100 for i in range(encoded_1['input_ids'].size(-1))]])
#         labels = torch.cat([labels, encoded_2['input_ids']], -1)
#         # print('labels', labels.size(), labels)
#         # print('input_ids', input_ids.size(), input_ids)
#
#         with torch.no_grad():
#             output = self.model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 labels=labels
#             )
#         # print(output.loss)
#         # exit(1)
#         perplexity = output.loss.numpy().tolist()
#
#         # print(perplexity)
#         acc_ppl = perplexity * encoded_2['input_ids'].size(-1)
#         num_tokens = encoded_2['input_ids'].size(-1)
#
#         return acc_ppl, num_tokens


class HumanStudyEvaluator:
    def __init__(self, data_file, article_titles, black_list, article_dir='data/evaluation'):
        self.data = {}
        self.black_list = black_list
        self.group_stats = {key: {
            'reading_time_all_text': [],
            'reading_time_question_text': [],
            'summary_length': [],
            'summary_time': [],
            'summary_bias': [],
            'summary_novel_uni_grams': [],
            'summary_novel_bi_grams': [],
            'summary_novel_tri_grams': [],
            'likert_question': {'relevance': [], 'timing': [], 'importance': []},
            'likert_experience': {'engaging': [], 'understanding': [], 'overall': []}
        } for key in ['control', 'authentic', 'generated']}

        self.group_articles = {
            'control': {},
            'authentic': {},
            'generated': {}
        }
        for group, group_dir in [('control', 'control'), ('authentic', 'authentic'), ('generated', 'prompt_gpt')]:
            for article_title in article_titles:
                article_file = os.path.join(article_dir, group_dir, 'articles', article_title+'.html')
                logging.info('loading article {} in group {} at {}'.format(article_title, group, article_file))
                with open(article_file, 'r') as fp:
                    article_page = ''.join(fp.readlines())
                article_parser = ArticleParser()
                article_parser.feed(article_page)
                # print(article_parser.paragraphs)
                article_parser.set_plain_paragraphs()

                self.group_articles[group][article_title] = article_parser

        self.__parse_data(data_file)
        '''
        {'user_id': user_id, 'group': group, 'phase_instruction': {}, 'phase_demographic': {}, 'phase_reading': {}, }
        '''

    def __parse_data(self, data_file):
        logging.info('parsing lines ...')
        num_lines = 0
        with open(data_file, 'r') as fp:
            for line in fp.readlines():
                num_lines += 1
                phase_data = json.loads(line.strip())
                if phase_data['prolific_pid'] in self.black_list:
                    continue
                uid_pid = '{}#{}'.format(phase_data['uid'], phase_data['prolific_pid'])
                if uid_pid not in self.data:
                    group_id = phase_data['uid'].split('_')[-1]
                    self.data[uid_pid] = {
                        'group': group_id,
                        'pid': phase_data['prolific_pid'],
                        'phase_ids': [],
                        'article_id': phase_data['aid'].split('#')[0],
                        'summary_info': None,
                        'reading_time_paragraph': [],
                        'reading_time_total': 0,
                        'reading_time_question_paragraphs_total': 0,
                        'demographic_info': {'age': None, 'gender': None, 'education': None, 'native_speaker': None, 'english_proficiency': None, 'reading_frequency': None},
                        'question_eval': {'relevance': [], 'timing': [], 'importance': []},
                        'meta_eval': {'engaging': 0, 'understanding': 0, 'overall': 0},
                        'timestamp': [0 for i in range(9 if group_id == 'control' else 9)],
                        'collections': []
                    }
                    assert self.data[uid_pid]['group'] in ['control', 'authentic', 'generated']
                self.data[uid_pid]['collections'].append(phase_data)

        logging.info('{} lines, {} data collected'.format(num_lines, len(self.data)))
        for uid_pid, phase_data in self.data.items():
            # print(phase_data)
            for item in phase_data['collections']:
                # print(item)
                # print('here', phase_data['group'], phase_data['timestamp'], len(phase_data['timestamp']), item['phase'])
                phase_data['phase_ids'].append([item['phase_start'], item['phase']])
                # continue
                phase_data['timestamp'][item['phase']] = item['phase_start']
                if item['phase'] == 1:
                    phase_data['demographic_info']['age'] = item['age#radio']
                    phase_data['demographic_info']['gender'] = item['gender#checkbox']
                    phase_data['demographic_info']['education'] = item['education#radio']
                    phase_data['demographic_info']['native_speaker'] = item['english_speaker#radio']
                    phase_data['demographic_info']['reading_frequency'] = item['read_often#radio']
                    phase_data['demographic_info']['english_proficiency'] = item['english_proficiency#radio']
                elif item['phase'] == 2:
                    for key in item:
                        if key.startswith('finish_reading_'):
                            phase_data['reading_time_paragraph'].append((int(key.split('_')[-1]), item[key]))
                    phase_data['reading_time_paragraph'].sort(key=lambda x: x[0])
                    phase_data['reading_time_paragraph'] = [_[1] for _ in phase_data['reading_time_paragraph']]
                elif item['phase'] == 3:
                    phase_data['summary_info'] = self.__analyze_summary(item['text_0'], phase_data['group'], phase_data['article_id'])
                    phase_data['summary'] = item['text_0']
                elif item['phase'] == 4 and phase_data['group'] != 'control':
                    phase_data['meta_eval']['engaging'] = int(item['likert_engaging#radio'])
                    phase_data['meta_eval']['understanding'] = int(item['likert_engaging#radio'])
                    phase_data['meta_eval']['overall'] = int(item['likert_overall#radio'])
                elif item['phase'] == 5:
                    for key in item:
                        if key.startswith('likert_'):
                            phase_data['question_eval']['relevance'].append((int(key.split('#')[0].split('_')[1]), item[key]))
                    phase_data['question_eval']['relevance'].sort(key=lambda x: x[0])
                    phase_data['question_eval']['relevance'] = [int(_[1]) for _ in phase_data['question_eval']['relevance']]
                elif item['phase'] == 6:
                    for key in item:
                        if key.startswith('likert_'):
                            phase_data['question_eval']['timing'].append((int(key.split('#')[0].split('_')[1]), item[key]))
                    phase_data['question_eval']['timing'].sort(key=lambda x: x[0])
                    phase_data['question_eval']['timing'] = [int(_[1]) for _ in phase_data['question_eval']['timing']]
                elif item['phase'] == 7 and phase_data['group'] != 'control':
                    for key in item:
                        # print(key)
                        if key.startswith('likert_'):
                            phase_data['question_eval']['importance'].append((int(key.split('#')[0].split('_')[1]), item[key]))
                    # print('here', phase_data['question_eval']['importance'])
                    phase_data['question_eval']['importance'].sort(key=lambda x: x[0])
                    phase_data['question_eval']['importance'] = [int(_[1]) for _ in phase_data['question_eval']['importance']]

            if phase_data['group'] == 'control':
                try:
                    assert len(phase_data['collections']) == 7
                except AssertionError:
                    logging.error('control group with {} phases'.format(len(phase_data['collections'])))
            elif phase_data['group'] in ['authentic', 'generated']:
                try:
                    assert len(phase_data['collections']) == 9
                except AssertionError:
                    logging.error('control group with {} phases'.format(len(phase_data['collections'])))

            # first_paragraph_time = phase_data['reading_time_paragraph'][0] - phase_data['timestamp'][2]
            # finish_0 is start_time
            phase_data['reading_time_paragraph'] = [phase_data['reading_time_paragraph'][i] - phase_data['reading_time_paragraph'][i-1] for i in range(1, len(phase_data['reading_time_paragraph']))]
            # phase_data['reading_time_paragraph'].insert(0, first_paragraph_time)
            # print('here', phase_data['summary_info'])
            phase_data['summary_info']['time'] = phase_data['timestamp'][4] - phase_data['timestamp'][3]
            phase_data['reading_time_total'] = sum(phase_data['reading_time_paragraph'])

            article = self.group_articles[phase_data['group']][phase_data['article_id']]
            try:
                # print('here', article.paragraphs_plain)
                assert len(article.paragraphs_plain) == len(phase_data['reading_time_paragraph'])
            except AssertionError:
                logging.error('group {}, article {}, num_paragraphs: {}, reading_time_paragraph: {}'.format(phase_data['group'], phase_data['article_id'], len(article.paragraphs_plain), len(phase_data['reading_time_paragraph'])))
            for idx in range(len(article.paragraphs)):
                if article.paragraphs[idx]['num_question'] > 0:
                    phase_data['reading_time_question_paragraphs_total'] += phase_data['reading_time_paragraph'][idx]

            phase_data.pop('collections')

        # for key, data in self.data.items():
        #     data['phase_ids'].sort(key=lambda x: x[0])
        #     data['phase_ids'] = [[to_date(item[0]/1000), item[1]] for item in data['phase_ids']]
        #     print(data['group'], data['pid'], data['article_id'], data['phase_ids'])
        # exit(1)

    def __analyze_summary(self, summary_text, group_id, article_id):
        # print(article_id)
        # print(self.group_articles)
        source_article = self.group_articles[group_id][article_id]
        source_article_text = ' '.join(source_article.paragraphs_plain)
        source_article_words = word_tokenize(source_article_text)

        summary_words = word_tokenize(summary_text)
        summary_length = len(summary_words)

        uni_gram_ratio = self.n_grams(source_article_words, summary_words, n=1)
        bi_gram_ratio = self.n_grams(source_article_words, summary_words, n=2)
        tri_gram_ratio = self.n_grams(source_article_words, summary_words, n=3)

        summary_info = {
            'text': summary_text,
            'words': summary_words,
            'length': summary_length,
            'time': 0,
            'novel_uni_gram_ratio': uni_gram_ratio,
            'novel_bi_gram_ratio': bi_gram_ratio,
            'novel_tri_gram_ratio': tri_gram_ratio
        }
        return summary_info

    def get_stats(self):
        for uip_pid, data in self.data.items():
            # print(self.group_stats[data['group']])
            self.group_stats[data['group']]['summary_length'].append(data['summary_info']['length'])
            self.group_stats[data['group']]['summary_time'].append(data['summary_info']['time'])
            self.group_stats[data['group']]['summary_novel_uni_grams'].append(data['summary_info']['novel_uni_gram_ratio'])
            self.group_stats[data['group']]['summary_novel_bi_grams'].append(data['summary_info']['novel_bi_gram_ratio'])
            self.group_stats[data['group']]['summary_novel_tri_grams'].append(data['summary_info']['novel_tri_gram_ratio'])
            self.group_stats[data['group']]['reading_time_all_text'].append(data['reading_time_total'])

            if data['group'] == 'control':
                continue

            self.group_stats[data['group']]['likert_question']['relevance'].extend(data['question_eval']['relevance'])
            self.group_stats[data['group']]['likert_question']['timing'].extend(data['question_eval']['timing'])
            self.group_stats[data['group']]['likert_question']['importance'].extend(data['question_eval']['importance'])

            self.group_stats[data['group']]['likert_experience']['engaging'].append(data['meta_eval']['engaging'])
            self.group_stats[data['group']]['likert_experience']['understanding'].append(data['meta_eval']['understanding'])
            self.group_stats[data['group']]['likert_experience']['overall'].append(data['meta_eval']['overall'])

            self.group_stats[data['group']]['reading_time_question_text'].append(data['reading_time_question_paragraphs_total'])

        for group in self.group_stats:
            print(info_line(group))
            pprint.pprint(self.group_stats[group])
            for key in self.group_stats[group]:
                if type(self.group_stats[group][key]) is list and self.group_stats[group][key]:
                    self.group_stats[group][key] = sum(self.group_stats[group][key]) / len(self.group_stats[group][key])
                    if 'time' in key:
                        self.group_stats[group][key] /= 60000
                elif type(self.group_stats[group][key]) is dict:
                    for score in self.group_stats[group][key]:
                        print('here', group, key, score)
                        if self.group_stats[group][key][score]:
                            self.group_stats[group][key][score] = sum(self.group_stats[group][key][score]) / len(self.group_stats[group][key][score])
        pprint.pprint(self.group_stats)
#
#
# def map_back(gen_file, ref_file, new_file):
#     record = {}  # {id: str, article: [], generated: []}
#     with open(ref_file, 'r') as fp:
#         for line in fp.readlines():
#             data = json.loads(line.strip())
#             if data['id'] not in record:
#                 record[data['id']] = {}
#             record[data['id']]['article'] = nltk.sent_tokenize(data['input'])
#             record[data['id']]['reference'] = data['output']
#
#     with open(gen_file, 'r') as fp:
#         for line in fp.readlines():
#             data = json.loads(line.strip())
#             record[data['id']]['generated'] = data['generated']
#
#     format_outputs = []
#
#     invalid_q = 0
#     total_q = 0
#     for data_id in record:
#         record[data_id]['format_output'] = []
#         items = record[data_id]['generated'].split(' [SEP] ')
#         total_q += len(items)
#         for item in items:
#             fields = item.split(' # ')
#             record[data_id]['format_output'].append({
#                 'raw': item,
#                 'pre_sentence': None,
#                 'type': None,
#                 'answer': None,
#                 'question': None,
#                 'question_id': None
#             })
#             # print(fields)
#             # exit(1)
#             if len(fields) != 4:
#                 invalid_q += 1
#                 continue
#             pre_index = fields[0].find('Previous Sentence: ')
#             question_index = fields[-1].find('Question: ')
#             answer_index = fields[2].find('Answer Keywords: ')
#             type_index = fields[1].find('Question Type: ')
#
#             if pre_index == -1 or question_index == -1 or answer_index == -1 or type_index == -1:
#                 invalid_q += 1
#                 continue
#             # position
#             record[data_id]['format_output'][-1]['pre_sentence'] = fields[0][pre_index+len('Previous Sentence: '):]
#             # type
#             record[data_id]['format_output'][-1]['type'] = fields[1][type_index+len('Question Type: ')]
#             # answer
#             record[data_id]['format_output'][-1]['answer'] = fields[2][answer_index+len('Answer Keywords: '):]
#             # question
#             record[data_id]['format_output'][-1]['question'] = fields[3][question_index+len('Question: '):]
#
#             # retrieve:
#             if record[data_id]['format_output'][-1]['pre_sentence'] in ['Introductory Question', 'Title Question'] :
#                 retrieved_pre_sent_id = -1
#             else:
#                 tokenized_query = nltk.word_tokenize(record[data_id]['format_output'][-1]['pre_sentence'])
#                 candidates = [nltk.word_tokenize(sentence) for sentence in record[data_id]['article']]
#                 bm25 = BM25Okapi(candidates)
#                 scores = bm25.get_scores(tokenized_query)
#                 sorted_idx = np.argsort(scores)
#                 retrieved_pre_sent_id = int(sorted_idx[-1])
#
#             # print(record[data_id]['format_output'][-1]['pre_sentence'], retrieved_pre_sent_id, record[data_id]['article'][retrieved_pre_sent_id])
#
#             record[data_id]['format_output'][-1]['question_id'] = retrieved_pre_sent_id + 1
#             format_outputs.append({
#                 'id': data_id,
#                 'article': record[data_id]['article'],
#                 'raw_output': record[data_id]['generated'],
#                 'format_output': record[data_id]['format_output']
#             })
#     print(len(format_outputs))
#     with open(new_file, 'w') as fp:
#         for output in format_outputs:
#             # print(type(output))
#             # for key in output:
#             #     print(key, type(key), type(output[key]))
#             fp.write(json.dumps(output)+'\n')
#
#     logging.info('total generated questions {}, {} invalid'.format(total_q, invalid_q))


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        # filename='log/{}_question_classification_{}.log'.format(args.job, args.dataset),  # 'kt_log_reg/train_kt_0.5_0.log',  # '{}.log'.format(args.job),  # args.kt_train_log,
        level=logging.INFO,
        filemode='w'
    )
    summary_evaluator = SummaryEvaluator(
        article_files=['classical_chinese_philosophy', 'introduction_to_socialization', 'the_brain_is_an_inference_machine', 'political_socialization_the_ways_people_become_political', 'sustainability_business_and_the_environment'],
        coherence_prompt_file='prompts/human_summary_evaluation/coherence_eval.txt',
        informativeness_prompt_file='prompts/human_summary_evaluation/informativeness_eval.txt',
        consistency_prompt_file='prompts/human_summary_evaluation/consistency_eval.txt',
        summary_file='results/human_study/human_study_data.jsonl',
        temperature=0,
        prompt_model='gpt-4o',
        user_blacklist=['5bc7a4433dec25000138c626'],
        score_output='summary_scores_t0_7_d.jsonl'
    )
    # summary_evaluator.get_average()

    pass
    # format_evaluation_data('classical_chinese_philosophy', 'data/evaluation/control/classical_chinese_philosophy', user_group='control', model_dir='control', base_dir='./data/evaluation')

    # format_evaluation_data('Sustainability: Business and the Environment', 'data/evaluation/authentic/Sustainability: Business and the Environment', user_group='authentic', model_dir='authentic', base_dir='./data/evaluation')
    # format_evaluation_data('Sustainability: Business and the Environment', 'data/evaluation/prompt_gpt/Sustainability: Business and the Environment', user_group='generated', model_dir='prompt_gpt', base_dir='./data/evaluation')
    # format_evaluation_data('Sustainability: Business and the Environment', 'data/evaluation/control/Sustainability: Business and the Environment', user_group='control', model_dir='control', base_dir='./data/evaluation')
    # all_articles = [
    #     # 'actors_in_the_international_system',
    #     # 'classical_chinese_philosophy',
    #     # 'elements_of_culture',
    #     # 'forms_of_business_organizations',
    #     # 'political_socialization_the_ways_people_become_political',
    #     #  'sustainability_business_and_the_environment',
    #     # 'the_brain_is_an_inference_machine',
    #     'introduction_to_socialization'
    # ]
    # for article_title in all_articles:
    #     print('building {}'.format(article_title))
    #     for group, model_dir in [('control', 'control'), ('authentic', 'authentic'), ('generated', 'prompt_gpt')]:
    #         format_evaluation_data(
    #             article_title=article_title,
    #             output_file='data/evaluation/{}/format/{}'.format(model_dir, article_title),
    #             user_group=group,
    #             model_dir=model_dir,
    #             post_test='summarization',
    #             base_dir='data/evaluation'
    #         )

    # HumanStudyEvaluator('experimental.jsonl')
    question_evaluator = QuestionEvaluator(output_file='openstax_output_bart_base.jsonl', reference_file='openstax_test.jsonl')
    # map_back(gen_file='results/openstax_output_bart_base.jsonl', ref_file='openstax_test.jsonl', new_file='results/openstax_output_bart_base_format.jsonl')
    # perplexity = Perplexity(model_name='gpt2-large')
    # perplexity.calculate(input_file='openstax_test_format.jsonl', key='format_output')


import copy

from openai import OpenAI
from torch.utils.data import Dataset
import yake
from config import local
import json
from common_utils import *
from tqdm import tqdm
import random
import nltk
import numpy as np
from rank_bm25 import BM25Okapi
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod


stopwords_en = set(nltk.corpus.stopwords.words('English'))


class DataBuilder:
    def __init__(self, data_file, prompt_file, temperature=0, max_request_cnt=3, prompt_model='gpt-3.5-turbo-1106', noise_rate=0.8, context_window=4, question_mark='[MASK]', sep_mark='[SEP]', num_kw=5):
        self.prompt_model = prompt_model
        self.temperature = temperature
        self.max_request_cnt = max_request_cnt
        with open(prompt_file, 'r') as fp:
            self.prompt = ''.join(fp.readlines())
        self.data_file = data_file
        self.noise_rate = noise_rate
        self.sep_mark = sep_mark
        self.question_mask = question_mark
        self.context_window = context_window
        self.client = OpenAI(api_key=local.openai_key)
        self.keyword_extractor = yake.KeywordExtractor(top=num_kw *2)
        self.num_keyword = num_kw
        self.kw_blacklist = ['article', 'discuss', 'address', 'suggest']

    def build(self, output_file, record_file, blacklist=None, test_data=None):
        output_fp = open(output_file, 'a')
        cache = set()
        with open(output_file, 'r') as fp:
            for line in fp.readlines():
                data = json.loads(line.strip())
                cache.add(data['id'])
        logging.info('{} cache data'.format(len(cache)))
        record_fp = open(record_file, 'a')

        with open(self.data_file, 'r') as fp:
            lines = fp.readlines()
        pbar_ = tqdm(
            lines,
            total=len(lines),
            desc='Preparing Training Data'
        )

        total_tokens = 0
        total_price = 0
        total_request = 0

        request_error_cnt = 0

        for line in pbar_:
            data = json.loads(line)
            if data['id'] in cache:
                logging.info('skip {} in cache'.format(data['id']))
                continue
            if blacklist and data['id'] in blacklist:
                continue
            # debug
            if test_data and data['id'] != test_data:
                continue
            questions_pos = [question['pos'] for question in data['questions']]
            logging.debug(info_line('Question Position Indicators'))
            logging.debug('Question Pos: {}'.format(questions_pos))
            pre_question_sentences = {}

            un_noise_set = set()

            for pos in questions_pos:
                un_noise_set.update([i for i in range(pos-self.context_window, pos+self.context_window)])
                if pos == -1:
                    pre_question_sentences[pos] = -1  # title question
                elif pos == 0:
                    pre_question_sentences[pos] = 0
                else:
                    pre_sentence_id = pos - 1
                    while pre_sentence_id in questions_pos:
                        pre_sentence_id -= 1  # continuous questions share the same indicator
                    pre_question_sentences[pos] = pre_sentence_id
            logging.debug('Position Indicators: {}'.format(pre_question_sentences))

            # step 1: remove questions
            processed_article = []  # Questions/Noise replaced with [MASK]s
            chunks = []  # question with context (for editing)
            logging.debug(info_line('Process Article with [Mask]'))
            noise_id = 0
            max_noise_cnt = int(self.noise_rate * len(data['questions']))
            for sid, sentence in enumerate(data['article']):
                random_factor = random.random()
                if sid in questions_pos:
                    processed_article.append(self.question_mask)
                    # mask_original_map[len(mask_original_map)] = [sid]
                    chunks.append([max(sid -self.context_window, 0), min(sid +self.context_window, len(data['article'] ) -1)])
                elif sid not in un_noise_set and noise_id < max_noise_cnt and random_factor < self.noise_rate:
                    logging.debug('noise removal: {}'.format(sid))
                    processed_article.append(self.question_mask)  # question indicator cannot be removed
                    chunks.append([max(sid -self.context_window, 0), min(sid +self.context_window, len(data['article'] ) -1)])
                    # mask_original_map[len(mask_original_map)] = [noise_id]
                    noise_id += 1
                else:
                    processed_article.append(sentence['text'])
            logging.debug('Processed article: {}'.format(' '.join(processed_article)))
            logging.debug('Raw chunks: {}'.format(chunks))

            # clear overlap
            new_chunks = []  # merge overlap question-context for prompting
            has_merge = True
            roun = 0
            while has_merge:
                roun += 1
                merge_cnt = 0
                chunk_id = 0
                while chunk_id < len(chunks):
                    if chunk_id < len(chunks ) -1 and chunks[chunk_id][1] >= chunks[chunk_id +1][0]:
                        new_chunks.append([
                            chunks[chunk_id][0],
                            chunks[chunk_id +1][1]
                        ])
                        if chunk_id == len(chunks) - 2:
                            break
                        chunk_id += 1
                        merge_cnt += 1
                    else:
                        new_chunks.append(chunks[chunk_id])
                    chunk_id += 1

                if merge_cnt > 0:
                    chunks = new_chunks
                    new_chunks = []
                else:
                    has_merge = False

            logging.debug('new chunks: {}'.format(new_chunks))

            paragraphs_to_edit = []
            mask_id = 0
            for start, end in new_chunks:
                para = []  # context + [MASK] + context
                for i in range(start, end +1):
                    if processed_article[i] == self.question_mask:
                        mask_id += 1
                        if len(para) > 0 and para[-1] == self.question_mask:
                            # mask_original_map[mask_id].append()
                            continue  # skip continuous [MASK]s
                    para.append(processed_article[i])
                paragraphs_to_edit.append(para)

            for para in paragraphs_to_edit:
                logging.debug('= ' *100)
                logging.debug('Prompt Input:'.format(' '.join(para)))

            logging.debug('~ ' *200)
            # step 2: replace [MASK]s with new sentences and make local edits
            logging.debug(info_line('Prompting'))
            edited_paragraphs = []
            success = False
            for pid, paragraph in enumerate(paragraphs_to_edit):
                logging.debug('= ' *100)
                prompt_content = self.prompt.format(input=' '.join(paragraph))
                messages = [{'role': 'user', 'content': prompt_content}]
                request_cnt = 0
                success = False
                logging.debug(info_line('{}/{} Paragraph of {}, sentences {}-{}, {} [MASK]'.format(
                    pid, len(paragraphs_to_edit), data['id'], new_chunks[pid][0], new_chunks[pid][1], paragraph.count(self.question_mask)
                )))
                logging.debug('Prompt: {}'.format(prompt_content))

                try:
                    response = self.client.chat.completions.create(
                        model=self.prompt_model,
                        messages=messages,
                        temperature=self.temperature
                    )
                    record = {
                        'id': data['id'],
                        'prompt': prompt_content,
                        'raw_response': response.choices[0].message.content,
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'prompt_cost': local.api_price[self.prompt_model]['prompt'] * response.usage.completion_tokens,
                        'completion_cost': local.api_price[self.prompt_model]['completion'] * response.usage.completion_tokens,
                        'request': request_cnt
                    }
                    record['total_cost'] = record['prompt_cost'] + record['completion_cost']
                    record['total_tokens'] = record['prompt_tokens'] + record['completion_tokens']
                    record_fp.write(json.dumps(record ) +'\n')
                    request_cnt += 1
                    total_tokens += record['total_tokens']
                    total_request += 1
                    total_price += record['total_cost']
                    logging.debug(info_line('', symbol='-'))
                    logging.debug('raw_output: {}'.format(response.choices[0].message.content))
                    parse_result = self.parse_response(response)  # rationale, edited_paragraph
                    if parse_result:
                        edited_paragraphs.append(parse_result[1])
                    success = True
                except Exception:
                    break

            if not success:
                request_error_cnt += 1
                logging.error('Example {} failed'.format(data['id']))
                continue

            assert len(edited_paragraphs) == len(new_chunks)
            logging.debug('~ ' * 200)
            edited_article = []
            pre_sent_id = 0

            for idx in range(len(edited_paragraphs)):
                for sid in range(pre_sent_id, new_chunks[idx][0]):
                    edited_article.append(data['article'][sid]['text'])
                edited_article.append(edited_paragraphs[idx])
                pre_sent_id = new_chunks[idx][1] + 1

            edited_article = ' '.join(edited_article)
            logging.debug(info_line('Input Article'))
            logging.debug(edited_article)

            edited_article_sentences = nltk.sent_tokenize(edited_article)
            candidates = [nltk.word_tokenize(sentence) for sentence in edited_article_sentences]

            logging.debug(info_line('Retrieve Question Position Indicator'))
            bm25 = BM25Okapi(candidates)
            for question_pos in pre_question_sentences:
                if question_pos == -1:
                    pre_question_sentences[question_pos] = 'Title Question'
                elif question_pos == 0:
                    pre_question_sentences[question_pos] = 'Introductory Question'
                else:
                    logging.debug(info_line('Question {}'.format(question_pos), symbol='-'))
                    original_indicator = data['article'][pre_question_sentences[question_pos]]['text']
                    logging.debug('Original indicator: {}'.format(original_indicator))
                    tokenized_query = nltk.word_tokenize(original_indicator)
                    candidate_scores = bm25.get_scores(tokenized_query)
                    sorted_idx = np.argsort(candidate_scores)
                    logging.debug('Ranking scores: {}'.format(candidate_scores))
                    logging.debug('Sorted ids: {}'.format(sorted_idx))
                    new_indicator = edited_article_sentences[sorted_idx[-1]]
                    pre_question_sentences[question_pos] = new_indicator
                    logging.debug('Sorted ids: {}'.format(new_indicator))
            logging.debug(info_line('Question Position Indicators'))
            logging.debug('New Position Indicators: {}'.format(pre_question_sentences))

            # step 3: construct output
            output_ = []
            for question in data['questions']:
                # Position Indicator: previous sentence
                position_indicator = pre_question_sentences[question['pos']]

                # Answer: top 5 keywords
                if question['answer'] == 'NO_ANSWER':
                    answer_keywords = 'No answer in the article'
                else:
                    answer_keywords = self.extract_keywords(question['answer'])

                # Complete Question
                complete_question = question['text']
                if question['complete'] != 'ALREADY_COMPLETE':
                    complete_question = question['complete']

                output_.append('# Previous Sentence: {} # Type: {} # Answer Keywords: {} # Question: {}'.format(
                    position_indicator, question['type']['question_type'].replace('_', ' '), answer_keywords, complete_question
                ))

            output_ = ' {} '.format(self.sep_mark).join(output_)
            logging.debug(info_line('Output Example'))
            output_fp.write(json.dumps({
                'id': data['id'], 'input': edited_article, 'output': output_
            } ) +'\n')

            pbar_.set_postfix(OrderedDict({
                'Tokens': total_tokens,
                'Price': total_price,
                'Request': total_request
            }))

        logging.info('Error examples {}'.format(request_error_cnt))

        record_fp.close()
        output_fp.close()

    def parse_response(self, response):
        try:
            raw_output = response.choices[0].message.content
            analysis = raw_output.split('Coherence Analyses: ')[-1].split('Edited Paragraph')[0].strip()
            edited_paragraph = raw_output.split('Edited Paragraph:')[-1].strip()
        except Exception:
            return
        return analysis, edited_paragraph

    def extract_keywords(self, answer, ):
        keywords = self.keyword_extractor.extract_keywords(answer)
        selected_keywords = []
        selected_words = []
        for kw, score in keywords:
            invalid = False
            for block_word in self.kw_blacklist:
                if block_word in kw:
                    invalid = True
                    break
            if invalid:
                continue
            repeat = False
            words = nltk.word_tokenize(kw)
            for word in words:
                if word in selected_words and word.lower() not in stopwords_en:
                    repeat = True
                    break
            if repeat:
                continue
            selected_keywords.append(kw)
            selected_words.extend(words)
            if len(selected_keywords) == self.num_keyword:
                break
        return ','.join(selected_keywords)


def my_collate(batch_data):
    new_batch_data = {}
    for key in batch_data[0].keys():
        if key != 'ids':
            new_batch_data[key] = torch.stack([data[key] for data in batch_data], dim=0)
        else:
            new_batch_data[key] = [data[key] for data in batch_data]
    return new_batch_data


class GuidingQDataset(Dataset):
    def __init__(self, data_file, tokenizer, prepare_decoder_input_ids_from_labels, max_input_length=2048, max_output_length=512):
        super(GuidingQDataset, self).__init__()
        self.tokenizer = tokenizer
        self.input_max_length = max_input_length
        self.output_max_length = max_output_length
        self.prepare_decoder_input_ids_from_labels = prepare_decoder_input_ids_from_labels

        self.ids = []
        self.x_encoder_input_ids = []
        self.x_encoder_attention_mask_ids = []
        self.y_decoder_label_ids = []
        self.y_decoder_input_ids = []

    @abstractmethod
    def __build_dataset(self, data_file):
        pass

    def __getitem__(self, idx):
        return {
            'ids': self.ids[idx],
            'x_encoder_input_ids': self.x_encoder_input_ids[idx],
            'x_encoder_attention_mask_ids': self.x_encoder_attention_mask_ids[idx],
            'y_decoder_label_ids': self.y_decoder_label_ids[idx],
            'y_decoder_input_ids': self.y_decoder_input_ids[idx]
        }

    def __len__(self):
        return len(self.ids)

    def locate_end(self, input_):
        if type(input_) == str:
            input_ = self.tokenizer(input_, padding='max_length', return_tensors='pt', max_length=self.input_max_length, truncation=True)
        truncated_article = self.tokenizer.batch_decode(input_['input_ids'])[0]
        truncated_sentences = nltk.sent_tokenize(truncated_article)
        return len(truncated_sentences) - 1


class End2EndDataset(GuidingQDataset):
    def __init__(self, data_file, tokenizer, prepare_decoder_input_ids_from_labels, output_position=True, output_type=True, output_answer=True, max_input_length=2048, max_output_length=512, add_instruction=True, truncate_questions=True, truncate_doc=True, doc_trunc_length=400, field_sep=' # ', entry_sep=' | ', max_examples=-1):
        self.output_position = output_position
        self.output_type = output_type
        self.output_answer = output_answer
        self.add_instruction = add_instruction
        self.truncate_questions = truncate_questions
        self.instructions = 'Generate questions for a given article. Each output is formulated as \"# Previous Sentence: the sentene precedes the question as the position indicator # Answer Keywords: keywords of the answer to the question # Question: generated question,\" and multiple outputs are separated by \"｜\". Input: {} Output:'
        super(End2EndDataset, self).__init__(data_file, tokenizer, prepare_decoder_input_ids_from_labels, max_input_length, max_output_length)
        self.truncate_doc = truncate_doc
        self.doc_trunc_length = doc_trunc_length
        self.field_sep = field_sep
        self.entry_sep = entry_sep
        self.max_examples = max_examples
        self.__build_dataset(data_file)

    def __build_examples(self, data_file):
        ids = []
        inputs_ = []
        outputs_ = []

        total_q = 0
        total_d = 0
        reserved_d = 0
        reserved_q = 0

        cnt = 0
        with open(data_file, 'r') as fp:
            for line in fp.readlines():
                if cnt > self.max_examples > 0:
                    break
                cnt += 1
                total_d += 1
                instance = json.loads(line.strip())

                if self.add_instruction:
                    input_ = self.instructions.format(instance['article'])
                else:
                    input_ = instance['article']
                encoded_input = self.tokenizer(input_, padding='max_length', return_tensors='pt', max_length=self.input_max_length, truncation=True)
                last_pos = self.locate_end(encoded_input)

                output_ = []

                total_q += len(instance['output'])
                # total_cnt += len(instance['output'])
                for idx, item in enumerate(instance['output']):
                    if item['position'] > last_pos and self.truncate_questions:
                        break
                    entry = []
                    if self.output_position:
                        entry.append('Previous Sentence: {}'.format(item['prev_sentence']))
                    if self.output_type:
                        entry.append('Type: {}'.format(item['type']))
                    if self.output_answer:
                        entry.append('Answer Keywords: {}'.format(item['answer']))
                    entry.append('Question: {}'.format(item['question']))
                    entry = ' # '.join(entry)
                    output_.append(entry)

                if len(output_) == 0:
                    continue
                reserved_q += len(output_)
                output_ = ' ｜ '.join(output_)

                ids.append(instance['id'])
                inputs_.append(input_)
                outputs_.append(output_)

        reserved_d = len(inputs_)

        logging.info('Total doc: {}, reserved doc: {}, total q: {}, reserved q: {}'.format(total_d, reserved_d, total_q, reserved_q))

        return ids, inputs_, outputs_

    def __build_examples_new(self, data_file):
        ids = []
        inputs_ = []
        outputs_ = []

        total_d_cnt = 0
        total_s_cnt = 0
        no_q_cnt = 0

        num_examples = 0
        with open(data_file, 'r') as fp:
            for line in tqdm(fp.readlines()):
                num_examples += 1
                if num_examples > self.max_examples > 0:
                    break
                total_d_cnt += 1
                instance = json.loads(line.strip())

                segments = []
                segment = {'start': -1, 'end': -1, 'text': []}
                acc_length = 0
                for sid, sentence in enumerate(instance['article_sentences']):
                    words = nltk.word_tokenize(sentence)
                    if len(words) + acc_length > self.doc_trunc_length:
                        segment['end'] = sid
                        segments.append(copy.deepcopy(segment))
                        acc_length = 0
                        segment['end'] = -1
                        segment['start'] = sid
                        segment['text'].clear()
                    segment['text'].append(sentence)
                    acc_length += len(words)

                if len(segment['text']) > 0:
                    segment['end'] = len(instance['article_sentences'])
                    segments.append(copy.deepcopy(segment))

                for seg_id, seg in enumerate(segments):
                    ids.append(instance['id'] + '#seg_' + str(seg_id))
                    input_ = ' '.join(seg['text'])
                    output_ = []
                    for item in instance['output']:
                        if seg['end'] > item['position'] >= seg['start']:
                            entry = []
                            if self.output_position:
                                entry.append('Previous Sentence: {}'.format(item['prev_sentence']))
                            if self.output_type:
                                entry.append('Type: {}'.format(item['type']))
                            if self.output_answer:
                                entry.append('Answer Keywords: {}'.format(item['answer']))
                            entry.append('Question: {}'.format(item['question']))
                            entry = self.field_sep.join(entry)
                            output_.append(entry)
                    if len(output_) == 0:
                        no_q_cnt += 1
                        output_ = 'No Question Needed [EDN]'
                    else:
                        output_ = self.entry_sep.join(output_) + ' [END]'
                        total_s_cnt += 1
                    inputs_.append(input_)
                    outputs_.append(output_)

        logging.info('total documents {}, total segments {}, no_q_segments {}'.format(total_d_cnt, total_s_cnt, no_q_cnt))
        return ids, inputs_, outputs_

    def __build_dataset(self, data_file):
        logging.info('Building end2end dataset with type {}, position {}, answer {}...'.format(self.output_type, self.output_position, self.output_answer))

        ids, inputs_, outputs_ = self.__build_examples_new(data_file)
        assert len(ids) == len(inputs_) == len(outputs_)

        for i in range(len(ids)):
            self.ids.append(ids[i])
            # input
            encoded_input = self.tokenizer(inputs_[i], padding='max_length', return_tensors='pt', max_length=self.output_max_length-10, truncation=True)
            self.x_encoder_input_ids.append(encoded_input['input_ids'].squeeze(0))
            self.x_encoder_attention_mask_ids.append(encoded_input['attention_mask'].squeeze(0))
            # output
            encoded_output = self.tokenizer(outputs_[i], padding='max_length', return_tensors='pt', max_length=self.output_max_length-10, truncation=True)
            # print('output', encoded_output)
            encoded_output['input_ids'][encoded_output['input_ids'] == self.tokenizer.pad_token_id] = -100
            self.y_decoder_label_ids.append(encoded_output['input_ids'].squeeze(0))
            self.y_decoder_input_ids.append(self.prepare_decoder_input_ids_from_labels(encoded_output['input_ids']).squeeze(0))

    def __getitem__(self, idx):
        return {
            'ids': self.ids[idx],
            'x_encoder_input_ids': self.x_encoder_input_ids[idx],
            'x_encoder_attention_mask_ids': self.x_encoder_attention_mask_ids[idx],
            'y_decoder_label_ids': self.y_decoder_label_ids[idx],
            'y_decoder_input_ids': self.y_decoder_input_ids[idx]
        }

    def __len__(self):
        return len(self.ids)


class PipelineDataset(GuidingQDataset, ABC):
    def __init__(self, data_file, tokenizer, prepare_decoder_input_ids_from_labels, task, max_input_length, max_output_length, add_instruction, truncate_data=True, doc_trunc_length=300):
        assert task in ['position_pred', 'answer_ext', 'question_gen']
        super(PipelineDataset, self).__init__(data_file, tokenizer, prepare_decoder_input_ids_from_labels, max_input_length, max_output_length)
        self.task = task
        self.truncate_data = truncate_data
        self.doc_trunc_length = doc_trunc_length
        self.instructions = {
            'position_pred': {'content':  'Given an article, select some sentences after which questions should be raised. [SEP] Article: {} [SEP] Sentences:', 'num_sentence': 2},
            'answer_ext': {'content': 'Identify keywords for a question that will be placed at a specific position of an article, marked by a [Question] symbol. [SEP] Article: {} [SEP] Keywords:', 'num_sentence': 4},
            'question_gen': {'content': 'Generate a question based on the article and keywords. The question will be placed at the [Question] position. [SEP] Article: {} [SEP] Keywords: {} [SEP] Question:', 'num_sentence': 2}
        }

        for key in self.instructions:
            self.instructions[key]['num_tokens'] = len(self.tokenizer(self.instructions[key]['content'])['input_ids'])

        self.add_instruction = add_instruction
        self.__build_dataset(data_file)

    def __build_dataset(self, data_file):
        ids = []
        inputs = []
        outputs = []
        if self.task == 'position_pred':
            with open(data_file, 'r') as fp:
                for line in fp.readlines():
                    instance = json.loads(line.strip())
                    segments = self.segmentation(instance)
                    for seg_id, seg in enumerate(segments):
                        ids.append(instance['id'] + '#pp' + '#seg_' + str(seg_id))
                        if self.add_instruction:
                            input_ = self.instructions['position_pred']['content'].format(' '.join(seg['text']))
                        else:
                            input_ = 'Article: {} [SEP] Sentences:'.format(' '.join(seg['text']))
                        output_ = []
                        for item in instance['output']:
                            if item['position'] == -1 and seg_id == 0:
                                output_.append(item['prev_sentence'])
                            if seg['end'] > item['position'] >= seg['start']:
                                output_.append(item['prev_sentence'])
                        if len(output_) == 0:
                            output_ = 'No Question Needed [END]'
                        else:
                            output_ = ' | '.join(output_) + ' [END]'

                        inputs.append(input_)
                        outputs.append(output_)

        elif self.task == 'answer_ext':
            with open(data_file, 'r') as fp:
                for line in fp.readlines():
                    instance = json.loads(line.strip())
                    segments = self.segmentation(instance)
                    for iid, item in enumerate(instance['output']):
                        for seg_id, seg in enumerate(segments):
                            if (item['position'] == -1 and seg_id == 0) or (seg['end'] > item['position'] >= seg['start']):
                                shift = item['position'] - seg['start'] + 1
                                # print('Here', shift, len(seg['text']), seg['start'], seg['end'])
                                seg['text'].insert(shift, '[Question]')
                                if self.add_instruction:
                                    input_ = self.instructions['answer_ext']['content'].format(' '.join(seg['text']))
                                else:
                                    input_ = 'Article: {} [SEP] Answer Keywords:'.format(' '.join(seg['text']))
                                output_ = item['answer'] + ' [END]'
                                ids.append(instance['id']+'#ae'+'_{}'.format(iid)+'#seg_{}'.format(seg_id))
                                inputs.append(input_)
                                outputs.append(output_)
                                seg['text'].pop(shift)
                                break

        elif self.task == 'question_gen':
            with open(data_file, 'r') as fp:
                for line in fp.readlines():
                    instance = json.loads(line.strip())
                    segments = self.segmentation(instance)

                    for iid, item in enumerate(instance['output']):
                        for seg_id, seg in enumerate(segments):
                            if (seg_id == 0 and item['position'] == -1) or (seg['end'] > item['position'] >= seg['start']):
                                shift = item['position'] - seg['start'] + 1
                                seg['text'].insert(shift, '[Question]')
                                if self.add_instruction:
                                    input_ = self.instructions['question_gen']['content'].format(' '.join(seg['text']), item['answer'])
                                else:
                                    input_ = 'Article: {} [SEP] Answer: {} [SEP] Questions:'.format(' '.join(seg['text']), item['answer'])
                                output_ = item['question'] + ' [END]'
                                ids.append(instance['id']+'#qg'+'_{}'.format(iid)+'#seg_{}'.format(seg_id))
                                inputs.append(input_)
                                outputs.append(output_)
                                seg['text'].pop(shift)
                                break

        assert len(ids) == len(inputs) == len(outputs)

        for idx in range(len(inputs)):
            self.ids.append(ids[idx])

            encoded_input = self.tokenizer(inputs[idx], padding='max_length', return_tensors='pt', max_length=self.input_max_length, truncation=True)
            self.x_encoder_input_ids.append(encoded_input['input_ids'].squeeze(0))
            self.x_encoder_attention_mask_ids.append(encoded_input['attention_mask'].squeeze(0))
            # output
            encoded_output = self.tokenizer(outputs[idx], padding='max_length', return_tensors='pt', max_length=self.output_max_length, truncation=True)
            encoded_output['input_ids'][encoded_output['input_ids'] == self.tokenizer.pad_token_id] = -100
            self.y_decoder_label_ids.append(encoded_output['input_ids'].squeeze(0))
            self.y_decoder_input_ids.append(self.prepare_decoder_input_ids_from_labels(encoded_output['input_ids']).squeeze(0))

    def segmentation(self, instance):
        segments = []
        segment = {'start': 0, 'end': -1, 'text': []}
        acc_length = 0
        for sid, sentence in enumerate(instance['article_sentences']):
            words = nltk.word_tokenize(sentence)
            if len(words) + acc_length > self.doc_trunc_length:
                segment['end'] = sid
                segments.append(copy.deepcopy(segment))
                acc_length = 0
                segment['end'] = -1
                segment['start'] = sid
                segment['text'].clear()
            segment['text'].append(sentence)
            acc_length += len(words)

        if len(segment['text']) > 0:
            segment['end'] = len(instance['article_sentences'])
            segments.append(copy.deepcopy(segment))
        return segments
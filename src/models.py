import json
import logging
import math
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup, BitsAndBytesConfig, Adafactor
from torch.utils.data import DataLoader, ConcatDataset
from peft import LoraConfig, get_peft_model, TaskType
from build_data import End2EndDataset, PipelineDataset, my_collate
from common_utils import find_between, merge_list, map_back
import torch
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from common_utils import parse_output


class QuestionGenerator:
    def __init__(self, model_name, task, max_input_length, max_output_length, model_type, output_type, output_answer, output_position):
        self.model_type = model_type
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.model_max_length = max_input_length
        assert task in ['position_pred', 'answer_ext', 'question_gen', 'all']
        assert self.model_type in ['end2end', 'pipeline', 'multitask']
        logging.info('Loading model {}'.format(model_name))

        self.model_type = model_type
        self.task = task
        self.output_type = output_type
        self.output_answer = output_answer
        self.output_position = output_position

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    def prepare_data(self, data_file, add_instruction, truncate_questions, max_examples=-1):
        dataset = None

        if self.model_type == 'end2end':
            dataset = End2EndDataset(
                data_file=data_file,
                tokenizer=self.tokenizer,
                prepare_decoder_input_ids_from_labels=self.model.prepare_decoder_input_ids_from_labels,
                output_position=self.output_position,
                output_type=self.output_type,
                output_answer=self.output_answer,
                max_input_length=self.max_input_length,
                max_output_length=self.max_output_length,
                add_instruction=add_instruction,
                truncate_questions=truncate_questions,
                max_examples=max_examples
            )

        elif self.model_type == 'pipeline':
            dataset = PipelineDataset(
                data_file=data_file,
                tokenizer=self.tokenizer,
                prepare_decoder_input_ids_from_labels=self.model.prepare_decoder_input_ids_from_labels,
                task=self.task,
                max_input_length=self.max_input_length,
                max_output_length=self.max_output_length,
                add_instruction=add_instruction,
                truncate_data=True
            )

        elif self.model_type == 'multitask':
            dataset = []
            for task in ['position_pred', 'answer_ext', 'question_gen']:
                dataset.append(PipelineDataset(
                    data_file=data_file,
                    tokenizer=self.tokenizer,
                    prepare_decoder_input_ids_from_labels=self.model.prepare_decoder_input_ids_from_labels,
                    task=task,
                    max_input_length=self.max_input_length,
                    max_output_length=self.max_output_length,
                    add_instruction=True
                ))
            dataset = ConcatDataset(dataset)
        return dataset

    def train(self, train_data_file, gpu_cnt, num_epoch, batch_size, warmup_rate, learning_rate, use_lora, model_save_dir, local_rank, add_instruction, truncate_questions, val_data_file, val_output_file, max_examples, data_name):

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        if use_lora:
            self.model = get_peft_model(self.model, lora_config)

        if gpu_cnt > 0:
            logging.info('Local rank {}, using {} GPU, id: {}, name {}'.format(local_rank, gpu_cnt, local_rank, torch.cuda.get_device_name(0)))

        if gpu_cnt == 0:
            device = torch.device('cpu')
        elif gpu_cnt == 1:
            device = torch.device('cuda:{}'.format(local_rank))
            self.model.to(device)

        else:
            dist.init_process_group(backend='nccl', init_method='env://')
            device = torch.device('cuda:{}'.format(local_rank))
            self.model.to(device)
            self.model = DDP(self.model, device_ids=[local_rank]).module
            batch_size = int(batch_size/gpu_cnt)

        logging.info('Local rank {}, loading {} {} dataset from {}'.format(local_rank, self.model_type, self.task, train_data_file))
        train_dataset = self.prepare_data(train_data_file, add_instruction=add_instruction, truncate_questions=truncate_questions, max_examples=max_examples)
        logging.info('Local rank {}, {} training examples in total.'.format(local_rank, len(train_dataset)))

        if gpu_cnt > 1:
            sampler = DistributedSampler(train_dataset)
            train_data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        else:
            train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # val_dataset = self.prepare_data(val_data_file, add_instruction=add_instruction, truncate_questions=truncate_questions, max_examples=-1)
        # val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        batch_steps = math.ceil(len(train_dataset) / batch_size / max(gpu_cnt, 1))
        total_steps = int(num_epoch) * batch_steps
        warmup_steps = int(total_steps * float(warmup_rate))

        # optimizer = AdamW(self.model.parameters(), lr=float(learning_rate))
        optimizer = Adafactor(
            self.model.parameters(),
            lr=learning_rate,
            scale_parameter=False,
            relative_step=False
        )
        # lr_scheduler = get_linear_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_warmup_steps=warmup_steps,
        #     num_training_steps=total_steps
        # )
        logging.info('Local rank: {}, Start train! data_size: {} batch_steps: {}, total_steps: {}, warmup_steps: {}'.format(local_rank, len(train_dataset), batch_steps, total_steps, warmup_steps))
        for epoch_id in range(num_epoch):
            self.model.train()
            epoch_loss = 0
            if gpu_cnt > 1:
                sampler.set_epoch(epoch_id)
            for batch_id, batch_data in enumerate(train_data_loader):
                outputs = self.model(
                    input_ids=batch_data['x_encoder_input_ids'].to(device),
                    attention_mask=batch_data['x_encoder_attention_mask_ids'].to(device),
                    labels=batch_data['y_decoder_label_ids'].to(device),
                    # decoder_input_ids=batch_data['y_decoder_input_ids'].to(device)
                )

                loss = outputs.loss
                loss.backward()
                if (batch_id+1) % 4 == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # lr_scheduler.step()
                logging.info('Local rank: {}, Model: {}, Task: {}, Epoch {}/{}, batch {}/{}, loss {}'.format(local_rank, self.model_type, self.task, epoch_id + 1, num_epoch, batch_id + 1, batch_steps, loss))
                epoch_loss += loss

            if len(train_data_loader) % 4 != 0:
                optimizer.step()
                optimizer.zero_grad()

            logging.info('Local rank: {}, Model: {}, Epoch {}/{} average loss: {}'.format(local_rank, self.model_name, epoch_id + 1, num_epoch, epoch_loss / batch_steps))

            if local_rank == 0:
                model_name = self.model_name.replace('/', '_')
                save_name = '{}_{}_{}_{}_'.format(data_name, model_name, self.model_type, self.task)
                if self.output_position:
                    save_name += 'p'
                if self.output_type:
                    save_name += 't'
                if self.output_answer:
                    save_name += 'a'
                save_name += '_{}.ckpt'.format(epoch_id)
                model_save_path = os.path.join(model_save_dir, save_name)

                logging.info('Local rank :{}, saving {} epoch model to {}'.format(local_rank, epoch_id, model_save_path))
                torch.save({
                    'epoch': num_epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, model_save_path)

        logging.info('Local rank {} finished training'.format(local_rank))

    def end2end_inference(self, model_save_path, data_loader, data_file, gpu_cnt, local_rank, output_file, max_examples=-1):
        if gpu_cnt == 0:
            device = torch.device('cpu')
        elif gpu_cnt == 1:
            device = torch.device('cuda:{}'.format(local_rank))
        else:
            device = torch.device('cuda:{}'.format(local_rank))

        if model_save_path:
            if gpu_cnt > 1:
                self.model = DDP(self.model).module

            logging.info('Loading model from {}, '.format(model_save_path))
            save_dict = torch.load(model_save_path)

            self.model.to(device)
            self.model.load_state_dict(save_dict['model_state_dict'])

        self.model.eval()
        if data_file:
            eval_dataset = self.prepare_data(data_file, truncate_questions=False, add_instruction=False, max_examples=-1)
            # print('='*100)
            data_loader = DataLoader(eval_dataset, batch_size=8, collate_fn=my_collate)

        logging.info('Start generation ...')
        num = 0
        output_fp = open(output_file, 'w')
        result = {}   # {id, article: [], article_sentences: [], output: [], }
        with torch.no_grad():
            for batch_id, data in enumerate(tqdm(data_loader)):
                outputs = self.model.generate(
                    input_ids=data['x_encoder_input_ids'].to(device),
                    attention_mask=data['x_encoder_attention_mask_ids'].to(device),
                    max_new_tokens=self.max_output_length,  # self.max_output_length
                    num_beams=4
                )
                # print('='*100)
                # print(self.tokenizer.batch_decode(data['x_encoder_input_ids'], skip_special_tokens=True))
                data['y_decoder_label_ids'][data['y_decoder_label_ids'] == -100] = self.tokenizer.pad_token_id
                # print(self.tokenizer.batch_decode(data['y_decoder_label_ids'], skip_special_tokens=True))
                outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for idx, output in enumerate(outputs):
                    instance_id = data['ids'][idx]
                    seg_id = instance_id.split('#')[-1]
                    if seg_id.startswith('seg_'):
                        seg_id = int(seg_id.split('_')[-1])
                    instance_id = '#'.join(instance_id.split('#')[:-1])
                    if instance_id not in result:
                        result[instance_id] = {
                            'id': instance_id,
                            'output': []
                        }
                    result[instance_id]['output'].append((seg_id, output))

                num += len(outputs)
                if num > max_examples > 0:
                    break

        for instance_id in result:
            has_q = False
            result[instance_id]['output'].sort(key=lambda x: x[0])
            raw = []
            merged_output = []
            for seg_id, seg_output in result[instance_id]['output']:
                raw.append(seg_output)
                if 'No Question Needed' in seg_output:
                    continue
                else:
                    merged_output.append(seg_output.replace('[END]', ''))
            # print('Here 2')
            result[instance_id]['output'] = ' | '.join(merged_output)
            result[instance_id]['output'] = parse_output(result[instance_id]['output'])

            # print('Here', result[instance_id]['output'])
            output_fp.write(json.dumps({
                'id': instance_id, 'output': result[instance_id]['output'], 'raw': raw
            })+'\n')

        # print('THere')
        # exit(1)
        output_fp.close()

    def pipeline_inference(self, data_file, pp_model_path, ae_model_path, qg_model_path, output_file, gpu_cnt, gpu_id):
        if gpu_cnt == 0:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(gpu_id))

        self.model.to(device)

        logging.info('Loading position prediction model from {}.'.format(pp_model_path))
        save_dict = torch.load(pp_model_path)
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.model.eval()

        # {'id': 'article_id', 'article': 'article', 'article_sentences': [], 'output': [{'position': 'pos', 'prev_sentence': 'prev_sent', 'answer': 'answer_keywords', 'type': 'question_type'}]}

        result = {}  # {'id': 'article_id, 'article': article, 'article_sentences': [], 'output': []}
        # Step 1: Position prediction
        position_pred_dataset = PipelineDataset(
            data_file,
            tokenizer=self.tokenizer,
            prepare_decoder_input_ids_from_labels=self.model.prepare_decoder_input_ids_from_labels,
            task='position_pred',
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            add_instruction=True,
        )
        position_pred_data_loader = DataLoader(position_pred_dataset, shuffle=False, batch_size=1)

        logging.info('Predicting Positions, {} data in total'.format(len(position_pred_dataset)))
        uniq_article_ids = set([])
        for batch_id, batch_data in tqdm(enumerate(position_pred_data_loader)):
            outputs = self.model.generate(
                input_ids=batch_data['x_encoder_input_ids'].to(device),
                attention_mask=batch_data['x_encoder_attention_mask_ids'].to(device),
                max_new_tokens=self.max_output_length
            )
            inputs = self.tokenizer.batch_decode(batch_data['x_encoder_input_ids'], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for idx, output in enumerate(outputs):
                id_fields = batch_data['ids'][idx].split('#')
                article_id = '#'.join(id_fields[:-2])
                uniq_article_ids.add(article_id)
                seg_id = int(id_fields[-1].split('_')[-1])
                if article_id not in result:
                    result[article_id] = {
                        'id': article_id,
                        'article': None,
                        'article_sentences': [],
                        'output': []
                    }

                result[article_id]['output'].append([seg_id, inputs[idx], output])
                # print('-'*100)
                # print('Here', batch_data['ids'][idx], output)
                # result[article_id] = {
                #     'id': article_id,
                #     'article': inputs[idx],
                #     'article_sentences': article_sentences,
                #     'output': [{'position': pre_ids[idx], 'pre_sentence': previous_sentences[idx]} for idx in range(len(previous_sentences))]
                # }
        no_q_article_ids = []
        for article_id in result:
            result[article_id]['output'].sort(key=lambda x: x[0])
            # print('old output', result[article_id]['output'])
            # print('1', result[article_id]['article'])
            # print('2', result[article_id]['article'])
            result[article_id]['article'] = []
            prev_sentences = []
            for seg_id, seg_input, seg_output in result[article_id]['output']:
                seg_text = find_between(seg_input, '[SEP] Article: ', '[SEP] Sentences:')
                result[article_id]['article'].append(seg_text)
                if 'No Questions Needed' in seg_output or 'No Question Needed' in seg_output:
                    continue
                seg_output = seg_output.replace('[END]', '').strip()
                prev_sentences.extend(seg_output.split('|'))
            result[article_id]['article'] = ' '.join(result[article_id]['article'])
            if len(prev_sentences) == 0:
                no_q_article_ids.append(article_id)
                continue

            article_sentences, positions = map_back(result[article_id]['article'], prev_sentences)
            result[article_id]['article_sentences'] = article_sentences
            new_outputs = [{'position': int(positions[i]), 'prev_sentence': prev_sentences[i], 'answer': 'answer_placeholder', 'type': 'type_placeholder', 'question': 'question_placeholder'} for i in range(len(prev_sentences))]
            # print('article:', result[article_id]['article'])
            # print('new_outputs', new_outputs)
            result[article_id]['output'] = new_outputs
            # print(result[article_id]['output'])
            # exit(1)

        for n_id in no_q_article_ids:
            result.pop(n_id)

        # for aid in result:
        #     print('='*100)
        #     print(aid, len(result[aid]['output']))
        #     for item in result[aid]['output']:
        #         print(item['position'])

        num_questions = sum([len(item['output']) for item in result.values()])
        logging.info('Finish position prediction, {}/{} articles with {} questions'.format(len(result), len(uniq_article_ids) , num_questions))
        # print(result)
        # exit(1)
        with open(output_file, 'w') as fp:
            for r in result:
                fp.write(json.dumps(result[r])+'\n')
        #
        # Step 2: Answer extraction
        if ae_model_path != pp_model_path:
            logging.info('Loading answer extraction model from {}'.format(ae_model_path))
            save_dict = torch.load(ae_model_path)
            self.model.load_state_dict(save_dict['model_state_dict'])

        answer_ext_dataset = PipelineDataset(
            output_file,
            tokenizer=self.tokenizer,
            prepare_decoder_input_ids_from_labels=self.model.prepare_decoder_input_ids_from_labels,
            task='answer_ext',
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            add_instruction=True
        )
        logging.info('Extracting Answers, {} data in total.'.format(len(answer_ext_dataset)))

        answer_ext_data_loader = DataLoader(answer_ext_dataset, shuffle=False, batch_size=1)
        for batch_id, batch_data in enumerate(answer_ext_data_loader):
            outputs = self.model.generate(
                input_ids=batch_data['x_encoder_input_ids'].to(device),
                attention_mask=batch_data['x_encoder_attention_mask_ids'].to(device),
                max_new_tokens=self.max_output_length
            )
            inputs = self.tokenizer.batch_decode(batch_data['x_encoder_input_ids'])
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for idx, output in enumerate(outputs):  # keyword for 1 question
                # print(info_line(batch_data['ids'][idx]))
                # print("input:", inputs[idx])
                # print('output:', output)
                # article_id = '#'.join(batch_data['ids'][idx].split('#')[:-2])
                # position_id = batch_data['ids'][idx].split('#')[-1]
                id_fields = batch_data['ids'][idx].split('#')
                # print(id_fields)

                article_id = '#'.join(id_fields[:-2])
                q_id = int(id_fields[-2].split('_')[1])
                # seg_id = int(id_fields[2].split('_')[1])
                result[article_id]['output'][q_id]['answer'] = output.replace('[END]', '').strip()
                # print('match', result[article_id]['output'][q_id])
        with open(output_file, 'w') as fp:
            for r in result:
                fp.write(json.dumps(result[r])+'\n')

        logging.info('Finish answer extracting.')
        # Step 3: Question Generation
        logging.info('Generating Questions...')
        if qg_model_path != ae_model_path:
            logging.info('Loading question generation model from {}'.format(qg_model_path))
            save_dict = torch.load(qg_model_path)
            self.model.load_state_dict(save_dict['model_state_dict'])

        question_gen_dataset = PipelineDataset(
            output_file,
            tokenizer=self.tokenizer,
            prepare_decoder_input_ids_from_labels=self.model.prepare_decoder_input_ids_from_labels,
            task='question_gen',
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            add_instruction=True
        )
        question_gen_dataloader = DataLoader(question_gen_dataset, batch_size=1, shuffle=False)
        for batch_id, batch_data in enumerate(question_gen_dataloader):
            qg_outputs = self.model.generate(
                input_ids=batch_data['x_encoder_input_ids'].to(device),
                attention_mask=batch_data['x_encoder_attention_mask_ids'].to(device),
                max_new_tokens=self.max_output_length
            )
            inputs = self.tokenizer.batch_decode(batch_data['x_encoder_input_ids'], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(qg_outputs)
            for idx, output in enumerate(outputs):
                id_fields = batch_data['ids'][idx].split('#')
                article_id = '#'.join(id_fields[:-2])
                q_id = int(id_fields[-2].split('_')[1])
                result[article_id]['output'][q_id]['question'] = output.replace('[END]', '').strip()
                # print('='*100)
                # print(batch_data['ids'][idx])
                # print('input:', inputs[idx])
                # print('output:', output)

        logging.info('Finish question generation.')
        with open(output_file, 'w') as fp:
            for r in result:
                fp.write(json.dumps(result[r])+'\n')


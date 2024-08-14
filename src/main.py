import logging
import os
import argparse
import torch.cuda

from config import local, euler
from models import QuestionGenerator
from evaluation import CrossEvaluator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--job', type=str,)
    parser.add_argument('--dataset', type=str, )
    parser.add_argument('--model_type', type=str,)  # end2end, pipeline, multitask
    parser.add_argument('--task', type=str)  # [question_pre, answer_ext, question_gen, all]
    parser.add_argument('--add_instruction', type=bool,)
    # for inference only
    parser.add_argument('--end2end_model_file', type=str)
    parser.add_argument('--multitask_model_file', type=str)
    parser.add_argument('--pipeline_pp_model_file', type=str,)
    parser.add_argument('--pipeline_ae_model_file', type=str,)
    parser.add_argument('--pipeline_qg_model_file', type=str, )
    parser.add_argument('--output_file', type=str)

    args = parser.parse_args()

    assert os.getenv('CUR_ENV') in ['local', 'euler']
    if os.getenv('CUR_ENV') == 'local':
        env_config = local
    else:
        env_config = euler

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        # filename='log/{data}_{job}_{model_type}_{task}.log'.format(data=args.dataset, job=args.job, model_type=args.model_type, task=args.task),
        level=logging.INFO,
        filemode='w'
    )

    if torch.cuda.device_count() > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0

    logging.info('Local rank {}, Running on {}'.format(local_rank, os.getenv('CUR_ENV')))

    if args.job == 'train':
        trainer = QuestionGenerator(
            model_name=args.model_name,
            task=args.task,
            output_type=True,
            output_answer=True,
            output_position=True,
            max_input_length=env_config.train_args['max_input_length'],
            max_output_length=env_config.train_args['max_output_length'],
            model_type=args.model_type
        )

        trainer.train(
            train_data_file=env_config.data[args.dataset]['train'],
            gpu_cnt=torch.cuda.device_count(),
            num_epoch=env_config.train_args['num_epoch'],
            batch_size=env_config.train_args['batch_size'],
            warmup_rate=env_config.train_args['warmup_rate'],
            learning_rate=env_config.train_args['learning_rate'],
            use_lora=False,  # env_config.train_args['use_lora'],
            model_save_dir=env_config.train_args['model_save_dir'],
            local_rank=local_rank,
            add_instruction=args.add_instruction,
            truncate_questions=True,
            max_examples=-1,
            val_data_file=env_config.data[args.dataset]['test'],
            val_output_file='val_output.jsonl',
            data_name=args.dataset
        )

    elif args.job == 'inference':
        trainer = QuestionGenerator(
            model_name=args.model_name,
            task=args.task,
            output_type=True,
            output_answer=True,
            output_position=True,
            max_input_length=env_config.train_args['max_input_length'],
            max_output_length=env_config.train_args['max_output_length'],
            model_type=env_config.train_args['model_type']
        )
        if args.model_type == 'end2end':
            trainer.end2end_inference(
                model_save_path=args.end2end_model_file,
                data_file=env_config.data[args.dataset]['test'],
                gpu_cnt=torch.cuda.device_count(),
                local_rank=local_rank,
                output_file=args.output_file,
                max_examples=-1,
                data_loader=None,
            )
        elif args.model_type == 'multitask':
            trainer.pipeline_inference(
                data_file=env_config.data[args.dataset]['test'],
                pp_model_path=os.path.join(env_config.train_args['model_save_dir'], args.multitask_model_file),
                ae_model_path=os.path.join(env_config.train_args['model_save_dir'], args.multitask_model_file),
                qg_model_path=os.path.join(env_config.train_args['model_save_dir'], args.multitask_model_file),
                output_file=args.output_file,
                gpu_cnt=1,
                gpu_id=0,
            )
        elif args.model_type == 'pipeline':
            trainer.pipeline_inference(
                data_file=env_config.data[args.dataset]['test'],
                pp_model_path=os.path.join(env_config.train_args['model_save_dir'], args.pipeline_pp_model_file),
                ae_model_path=os.path.join(env_config.train_args['model_save_dir'], args.pipeline_ae_model_file),
                qg_model_path=os.path.join(env_config.train_args['model_save_dir'], args.pipeline_qg_model_file),
                output_file=args.output_file,
                gpu_cnt=1,
                gpu_id=0
            )
    elif args.job == 'eval':
        q_evaluator = QuestionEvaluator(
            ppl_model_name=env_config.eval_args['ppl_model_name'],
            output_file=args.output_file,
            article_file=env_config.data[args.dataset]['test'],
            cal_ppl=False,
            cal_num=True,
            cal_dist=True,
            cal_rel=False,
            rel_model_name='paraphrase-MiniLM-L6-v2'
        )

        q_evaluator.eval()
        cross_evaluator = CrossEvaluator(
            output_file=args.output_file,
            reference_file=env_config.data[args.dataset]['test'],
            b_score=True
        )
        cross_evaluator.cross_eval()
    else:
        logging.error('unknown job')
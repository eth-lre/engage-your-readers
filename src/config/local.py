

api_price = {
    'gpt-4-1106-preview': {'prompt': 0.01/1000, 'completion': 0.03/1000},
    'gpt-4': {'prompt': 0.03/1000, 'completion': 0.06/1000},
    'gpt-4-32k': {'prompt': 0.06/1000, 'completion': 0.12/1000},
    'gpt-3.5-turbo-1106': {'prompt': 0.001/1000, 'completion': 0.002/1000}
}

data = {
    'arxiv': {
        'path': 'data/dataset/arxiv.jsonl',
        'min_article_length': 100,
        'max_article_length': 3000,
        'min_question_num': 3,
        'max_question_num': 25,
    },
    'pubmed': {
        'path': 'data/dataset/pubmed.jsonl',
        'min_article_length': 100,
        'max_article_length': 3000,
        'min_question_num': 3,
        'max_question_num': 25,
    },
    'textbook': {
        'test': 'data/dataset/splits/openstax_test_format.jsonl',
        'train': 'data/dataset/splits/openstax_train_format.jsonl',
        'min_article_length': 100,
        'max_article_length': -1,
        'min_question_num': 3,
        'max_question_num': 25,
    },
    'sci': {
        'test': 'data/dataset/splits/sci_test_format.jsonl',
        'train': 'data/dataset/splits/sci_train_format.jsonl'
    }
}

annotation_config = {
    'annotation_dir': 'data/annotation',
    'best_model': 'gpt-4-1106-preview',
    'question_verification': {
        'model': 'gpt-3.5-turbo-1106',
        'max_try': 1,
        'batch_size': 20,
        'temperature': 0,
        'prompt_template': 'prompts/s1_question_verification',
        'req_time_interval': 0.05,
        'output': 'data/annotation/question_verification/request_record_{}.jsonl'
    },
    'question_completion': {
        'model': 'gpt-3.5-turbo-1106',
        'max_try': 1,
        'batch_size': 10,
        'temperature': 0,
        'prompt_template': 'prompts/s2_question_completion',
        'req_time_interval': 0.05,
        'output': 'data/annotation/question_completion/request_record_{}.jsonl'
    },
    'answer_generation': {
        'model': 'gpt-3.5-turbo-1106',
        'max_try': 1,
        'batch_size': 1,
        'temperature': 0,
        'prompt_template': 'prompts/s3_answer_generation',
        'req_time_interval': 0.05,
        'output': 'data/annotation/answer_generation/request_record_{}.jsonl'
    },
    'evidence_extraction': {
        'beta': 1,
        'lowercase': True,
        'forward_search': True,
        'output': 'data/annotation/evidence_extraction/request_record_{}.jsonl'
    },
    'question_classification': {
        'method': 'rule',
        'model': 'gpt-3.5-turbo-1106',
        'max_try': 1,
        'batch_size': 1,
        'temperature': 0,
        'prompt_template': 'prompts/s5_question_classification',
        'req_time_interval': 0.05,
        'output': 'data/annotation/question_classification/request_record_{}.jsonl'
    }
}

eval_config = {
    'human_study_result_file': 'new_data.jsonl'
}


eval_args = {
    'ppl_model_name': 'gpt2-large'
}

train_args = {
    'model_name': 'google/flan-t5-large', # 't5-base',
    'model_type': 'end2end',
    'batch_size': 2,
    'num_epoch': 10,
    'learning_rate': 5e-5,
    'warmup_rate': 0.02,
    'max_input_length': 512,
    'max_output_length': 512,
    'model_save_dir': '/cluster/project/sachan/pencui/ProjectsData/GuidingQ/ckpts',
    'use_lora': False
}

data = {
    'textbook': {
        'train': '/cluster/project/sachan/pencui/ProjectsData/GuidingQ/data/openstax_train_format.jsonl',
        'test': '/cluster/project/sachan/pencui/ProjectsData/GuidingQ/data/openstax_test_format.jsonl',
    },
    'sci': {
        'train': '/cluster/project/sachan/pencui/ProjectsData/GuidingQ/data/sci_train_format.jsonl',
        'test': '/cluster/project/sachan/pencui/ProjectsData/GuidingQ/data/sci_test_format.jsonl'
    }
}

eval_args = {
    'ppl_model_name': 'gpt2-large'
}
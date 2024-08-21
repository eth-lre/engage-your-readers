# Questions for Active Reading

#### Code and Data for **[How to Engage Your Reader? Generating Guiding Questions to Promote Active Reading](https://aclanthology.org/2024.acl-long.632/)**

Peng Cui, Xiaoyu Zhang, Vilém Zouhar, and Mrinmaya Sachan. 

[![](https://img.shields.io/badge/License-MIT-blue.svg)]()

---
## Introduction
We study how human writers use questions in academic writing and how these questions influence human reading comprehension.
To this end, we
1. curate GuidingQ, a dataset of in-text questions in textbooks from [Openstax](https://openstax.org/) and research articles from [arXiv](https://arxiv.org/) and [Pubmed](https://pubmed.ncbi.nlm.nih.gov/).
2. fine-tune various LMs to generate these questions to promote active reading.   
3. conduct a human study to investigate the effect of such questions. 

---
## Dataset
Our dataset is available at https://drive.google.com/file/d/1BlMUkQAt95pL8V4Mj_Y9I-X8FPCiWNl9/view?usp=drive_link.

Each example consists of 
```
{
    "id": "article_id",
    "title": "article_title",
    "article": "list of article sentences",
    "questions": "list of extracted questions and their information"
}
```
---
## Experiment
+ Code for model training and evaluation: [src](https://github.com/eth-lre/engage-your-readers/tree/main/src).
+ Usage: [src/main](https://github.com/eth-lre/engage-your-readers/blob/main/src/main.py).
+ Model ckpt: [finetuned models]().

---
## Human Study

We release our human study UI for similar research in the future. See a [demo version](https://vilda.net/s/reading-comprehension-help/?uid=demo_paper&phase=2).

![image](https://github.com/zouharvi/reading-comprehension-help/assets/7661193/a1c8d0d5-5327-4b53-8f71-a034a2d53347)


Screen pipeline:
1. Demographic questions
2. Intructions
3. Reading task
4. Performance questions
5. Project questions
6. Reading interface (again) with question annotation
7. Exit

### Running this Experiment Interface

First you need to start the logger locally:
```
git clone https://github.com/zouharvi/annotation-logger.git
cd annotation-logger
# run it in the background
nohup python3 main.py &
```

Then, build 

```
git clone https://github.com/zouharvi/reading-comprehension-help
cd reading-comprehension-help/annotation_ui
npm install
# run locally and use local logger, not the live one
npm run dev
# go to this url which loads the file `annotation_ui/web/queues/demo_authentic.jsonl`
xdg-open localhost:9001?uid=demo_authentic
```

You can go from here by adding new user queues to `annotation_ui/web/queues` and loading them with the `uid=` parameter.

## Cite
```
@inproceedings{cui-etal-2024-engage,
    title = "How to Engage your Readers? Generating Guiding Questions to Promote Active Reading",
    author = "Cui, Peng  and
      Zouhar, Vil{\'e}m  and
      Zhang, Xiaoyu  and
      Sachan, Mrinmaya",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.632",
    pages = "11749--11765"
}
```

## Contact
For any questions, feel free to open an issue or drop me an email at peng.cui@inf.ethz.ch. 
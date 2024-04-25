# Human Reading Comprehension Help

Description TODO. See a [demo version](https://vilda.net/s/reading-comprehension-help/?uid=demo_paper&phase=2).

WIP project at ETH Zürich: Peng Cui, Xiaoyu Zhang, Vilém Zouhar, Mrinmaya Sachan and others.

![image](https://github.com/zouharvi/reading-comprehension-help/assets/7661193/a1c8d0d5-5327-4b53-8f71-a034a2d53347)


Screen pipeline:
1. Demographic questions
2. Intructions
3. Reading task
4. Performance questions
5. Project questions
6. Reading interface (again) with question annotation
7. Exit

## Running this Experiment Interface

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

# NLP-Chinese
天池竞赛 <NLP中文预训练模型泛化能力挑战赛> 解决方案。

## 方案简介
### 模型结构
模型整体上为简单的硬共享模式，


## 环境配置

### 训练环境
|项|值|
|:-:|:-:|
|操作系统|Ubuntu 18.04.4 LTS (Bionic Beaver)|
|GPU型号|NVIDIA Tesla P100 PCIe 16 GB|
|GPU驱动版本|450.51.06|
|CUDA 版本|11.0|
|Python版本|3.7.6|

### 安装依赖
推荐使用Conda安装所需依赖。

1. 从[官网][1]下载安装Miniconda或者Anaconda；
2. 运行以下命令从我们提供的 `environment.yml` 文件创建一个名为 `nlpc` 的虚拟环境并安装所有需要的依赖项：
```
conda env create -f environment.yml
```
3. 如果第2步运行成功，请忽悠这一步。如果失败，也可以逐行运行以下命令手动创建并安装依赖项：
```
conda create -n nlpc python=3.7.6
conda activate nlpc
conda install pytorch=1.7.0 torchvision torchtext cudatoolkit=11.0 -c pytorch
pip install transformers==3.5.1
conda install notebook pandas matplotlib scikit-learn flake8 pyyaml
pip install tensorboardx
conda install tensorboard
```

4. 激活新建的虚拟环境（如果还未激活）：
```
conda activate nlpc
```

5. 进入 code 文件夹以保证下面的命令按预期运行：
```
cd code
```

## 数据预处理
运行脚本 `preprocess.py` 会完成所有的数据预处理：
```
python preprocess.py \
    --input-tc-dirpath ../tcdata/ \
    --input-additional-dirpath ../user_data/additional_data/ \
    --output-dirpath ../user_data/preprocessed_data/
```
 `--input-tc-dirpath` 需指定为天池提供的数据所在的文件夹路径。

`--input-additional-dirpath` 为额外的公开数据集所在的文件夹路径。这些额外的数据包括：
1. CLUE 官方提供的 [OCNLI][2]，[CMNLI][3] 和 [TNEWS][4] 这三个任务的公开数据集；
2. 公开可获取的关于【今日头条新闻标题分类的数据集】。

`--output-dirpath` 为预处理后的文件的存放文件夹路径。

所执行的预处理包括：
- 从CLUE公开的 OCNLI，CMNLI 和 TNEWS 任务的 json 文件获取得到对应的训练集和验证集的 csv 文件并去掉标签为空的样本；
- 将所有的全角标点符号转换成半角标点符号；
- 将额外的今日头条新闻标题分类数据集和竞赛 TNEWS 数据集在一起做去重操作，确保额外的数据和竞赛数据（包括训练集，验证集和测试集）没有重复且竞赛数据自身没有重复。
- 使用 `emojiswitch` 库将 OCEMOTION 数据中的不在预训练 Transformer 词汇表中的表情符号转换为对应的文字描述，例如："😭" -> ":大声哭喊的脸:"；
- 将 OCEMOTION 数据中连续重复出现的超长字符串替换成较短的字符串，例如："[怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒]" -> "[怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒][怒]"。

## 训练流程复现
### 数据集划分
对于 OCNLI 和 TNEWS 任务，因为 CLUE 原本的的数据已经切分好了训练集与验证集（训练集和验证集为分开的 `*_train.json` 和 `*_dev.json` 文件），考虑到官方划分的验证集更均衡，因此这两个任务直接使用官方的划分结果来切分训练集和验证集。对于 OCEMOTION 任务，随机划分20%作为验证集，其余作为训练集。

### 在 CMNLI 数据集上预训练 OCNLI 单任务模型
这一步使用 CMNLI 的训练集加验证集为训练集，以 OCNLI 的验证集为验证集来预训练 OCNLI 单任务模型，使用的配置文件为 `roberta-large-first-ocnli-pre-ce-uni.yml`。

运行以下命令执行这一步训练：
```
python train.py \
    --config ../user_data/configs/roberta-large-first-ocnli-pre-ce-uni.yml \
    --gpu-ids 0 \
    --save-dirpath ../user_data/checkpoints/retrain/ocnli_pre/
```
训练集样本数大约40万，以32的有效批样本数（因显存限制，梯度累计8步，实际批样本为4）训练2万步（接近但不到2个epoch），每1万步做一次验证，保存验证集最优的模型权重文件 `checkpoint.pth` 至 `--save-dirpath` 指定的文件夹。训练在单个 NVIDIA Tesla P100 GPU 上大约耗时8.5小时。

### 在 OCNLI 数据集上继续预训练 OCNLI 单任务模型
这一步首先加载上一步保存的模型权重，然后以 OCNLI 的训练集为训练集，以 OCNLI的验证集为验证集来继续预训练 OCNLI 单任务模型，使用的配置文件为 `roberta-large-first-ocnli-ce-uni.yml`

运行以下命令执行这一步训练：
```
python train.py \
    --config ../user_data/configs/roberta-large-first-ocnli-ce-uni.yml \
    --gpu-ids 0 \
    --load-pthpath ../user_data/checkpoints/retrain/ocnli_pre/checkpoint.pth \
    --save-dirpath ../user_data/checkpoints/retrain/ocnli/
```
应指定 `--load-pthpath` 命令行参数来加载上一步保存的 checkpoint。训练集样本数5万左右，以32的有效批样本数（因显存限制，梯度累计8步，实际批样本为4）训练2个epoch，每个epoch结束做一次验证，保存验证集最优的模型权重文件 `checkpoint.pth` 至 `--save-dirpath` 指定的文件夹。训练在单个 NVIDIA Tesla P100 GPU 上大约耗时1小时。

### 在 TNEWS 额外数据集上预训练 TNEWS 单任务模型
这一步以处理后的今日头条标题分类额外数据集为训练集，以 TNEWS 验证集为验证集来预训练 TNEWS 单任务模型，使用的配置文件为 `roberta-large-first-tnews-pre-ce-uni.yml`。

运行以下命令执行这一步训练：
```
python train.py \
    --config ../user_data/configs/roberta-large-first-tnews-pre-ce-uni.yml \
    --gpu-ids 0 \
    --save-dirpath ../user_data/checkpoints/retrain/tnews_pre/
```
训练样本数大约23.5万，以32的有效批样本数（因显存限制，梯度累计8步，实际样本为4）训练2个epoch，每个epoch结束做一次验证，保存验证集最优的模型权重文件 `checkpoint.pth` 至 `--save-dirpath` 指定的文件夹。训练在单个 NVIDIA Tesla P100 GPU 上大约耗时3.5小时。

### 在 TNEWS 数据集上继续预训练 TNEWS 单任务模型
这一步首先加载上一步保存的模型权重，然后以 TNEWS 训练集为训练集，以 TNEWS 验证集为验证集来继续预训练 TNEWS 单任务模型，使用的配置文件为 `roberta-large-first-tnews-ce-uni.yml`。

运行以下命令执行这一步训练：
```
python train.py \
    --config ../user_data/configs/roberta-large-first-tnews-ce-uni.yml \
    --gpu-ids 0 \
    --load-pthpath ../user_data/checkpoints/retrain/tnews_pre/checkpoint.pth \
    --save-dirpath ../user_data/checkpoints/retrain/tnews/
```
训练样本数大约4.8万，以32的有效批样本数（因显存限制，梯度累计8步，实际样本为4）训练2个epoch，每个epoch结束做一次验证，保存验证集最优的模型权重文件 `checkpoint.pth` 至 `--save-dirpath` 指定的文件夹。训练在单个 NVIDIA Tesla P100 GPU 上大约耗时50分钟。

### 在 OCEMOTION 数据集上预训练 OCEMOTION 单任务模型
因为 OCEMOTION 没有使用相关的额外数据集，因此直接在竞赛数据集上预训练相应的单任务模型。80%数据作为训练集，20%数据作为验证集，使用的配置文件为 `roberta-large-first-ocemotion-ce-uni.yml`。

运行以下命令执行这一步训练：
```
python train.py \
    --config ../user_data/configs/roberta-large-first-ocemotion-ce-uni.yml \
    --gpu-ids 0 \
    --save-dirpath ../user_data/checkpoints/retrain/ocemotion/
```
训练样本数大约2.8万，以32的有效批样本数（因显存限制，梯度累计16步，实际样本为2）训练2个epoch，每个epoch结束做一次验证，保存验证集最优的模型权重文件 `checkpoint.pth` 至 `--save-dirpath` 指定的文件夹。训练在单个 NVIDIA Tesla P100 GPU 上大约耗时1小时。

### 参数平均与融合
脚本 `param_avg.py` 用来读取三个单任务模型的权重文件，将共享部分的参数取均值并融合其余参数组成硬共享模式需要的新的模型权重文件。

运行以下命令执行这一过程：
```
!python param_avg.py \
    --input-pthpaths ../user_data/checkpoints/retrain/ocnli/checkpoint.pth \
    ../user_data/checkpoints/retrain/ocemotion/checkpoint.pth \
    ../user_data/checkpoints/retrain/tnews/checkpoint.pth \
    --output-pthpath ../user_data/checkpoints/retrain/checkpoint_averaged.pth
```
`input-pthpaths` 接收多个路径，为保证生成的新的模型权重文件可以正确的加载，请确保按照 `ocnli`，`ocemotion`，`tnews` 这样的顺序传入对应的 `checkpoint.pth` 文件路径。

`--output-pthpath` 为新生成的模型权重 checkpoint 文件的路径。

### 在竞赛数据集上 Fine-Tune 多任务模型
这一步将加载上一步平均与融合之后的模型权重为初始参数，在竞赛提供的所有三个数据集上 Fine-Tune 最终的多任务模型。

运行以下命令执行这一步训练：
```
python train.py \
    --config roberta-large-first-hard-ce-uni.yml \
    --gpu-ids 0 \
    --load-pthpath ../user_data/checkpoints/retrain/checkpoint_averaged.pth \
    --save-dirpath ../user_data/checkpoints/retrain/checkpoint.pth
```



## 预测
The evaluate script simply load the model checkpoint specified by `--load-pthpath` and report the validation loss and score.
```
python evaluate.py \
    --config checkpoints/run1/config.yml \
    --gpu-ids 0 \
    --load-pthpath checkpoints/run1/checkpoint.pth \
```

## Predict
The predict script load the model checkpoint specified by `--load-pthpath` and predict on testset. Then, write the predict results to a `.zip` file which could directly upload to TianChi evaluate server.
```
python predict.py \
    --config checkpoints/run1/config.yml \
    --gup-ids 0 \
    --load-pthpath checkpoints/run1/checkpoint.pth \
    --save-zippath submissions/submission.zip
```

[1]: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
[2]: https://storage.googleapis.com/cluebenchmark/tasks/ocnli_public.zip
[3]: https://storage.googleapis.com/cluebenchmark/tasks/cmnli_public.zip
[4]: https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip
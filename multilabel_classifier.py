import os
import wandb
import pandas as pd
import ast
import numpy as np
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

if "FC_Topic_Classification" not in os.getcwd():
    os.chdir("FC_Topic_Classification")

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def compute_metrics(x):
    print('compute metrics')
    epoch = str(trainer.state.epoch).replace(".", "_")

    predictions = (x.predictions > 0) * 1
    labels = x.label_ids

    metrics = {}

    for idx, cat in enumerate(mlb.classes_):
        acc = accuracy_score(labels[:, idx], predictions[:, idx])
        pre = precision_score(labels[:, idx], predictions[:, idx])
        re = recall_score(labels[:, idx], predictions[:, idx])
        metrics.update({f'{cat}_acc': acc,
                        f'{cat}_prec': pre,
                        f'{cat}_rec': re,
                        })

        eval_data[f'{cat}_pred'] = predictions[:, idx]
        eval_data[f'{cat}_score'] = sigmoid(x.predictions[:, idx])

    eval_data.to_csv(f'{output_dir}/eval_data_epoch{epoch}.tsv', index=False, sep='\t')

    return metrics


if __name__ == '__main__':
    num_gpus = len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))

    model_id = 'roberta-base'

    model_configs = {'roberta-base': {'name': 'roberta-base',
                                         'tokenizer_config': {'pretrained_model_name_or_path': 'roberta-base',
                                                              'max_len': 512},
                                        'dataloader_config':{'per_device_train_batch_size': 8,
                                                             'per_device_eval_batch_size': 8}}
                     }

    config = model_configs[model_id]

    dataloader_config = config['dataloader_config']

    tokenizer_config = config['tokenizer_config']

    wandb.init(project="FC_Topic_Clf",
               name="",
               tags=[config['name']])

    wandb.config.update(dataloader_config)
    wandb.config.update(tokenizer_config)
    wandb.config.model_name = config['name']

    output_dir = f'FC_Topic_Clf/{wandb.run.name}'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    url2cat_df = pd.read_pickle('fc_claims_with_cat.pkl')
    #url2cat_df['categories'] = url2cat_df['categories'].apply(lambda x: ast.literal_eval(x))

    mlb = MultiLabelBinarizer()
    mlb = mlb.fit(url2cat_df['categories'].tolist())

    train_data = url2cat_df.sample(frac=0.8, random_state=0, replace=False).reset_index(drop=True)
    #train_data_other = train_data[train_data['categories'].apply(lambda x: x == ['Other'])]
    #train_data_other = train_data_other.sample(n=200, random_state=0)
    #train_data_not_other = train_data[train_data['categories'].apply(lambda x: x != ['Other'])]
    #train_data = pd.concat([train_data_other, train_data_not_other])
    train_labels = mlb.transform(train_data['categories'].tolist()) * 1.0
    train_data['label'] = train_labels.tolist()

    eval_data = url2cat_df[~url2cat_df['url'].isin(train_data['url'])]
    eval_labels = mlb.transform(eval_data['categories'].tolist()) * 1.0
    eval_data['label'] = eval_labels.tolist()

    print('load model and tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
    tokenizer.save_pretrained("tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(config['name'], num_labels=len(mlb.classes_), problem_type='multi_label_classification').cuda()

    train_data["text"] = train_data["text"].str.lower()
    eval_data["text"] = eval_data["text"].str.lower()

    print('transform data to Datasets')#
    train_dataset = Dataset.from_pandas(train_data[['text', 'label']])
    eval_dataset = Dataset.from_pandas(eval_data[['text', 'label']])

    print('tokenize tweets')
    def tokenize_function(examples):
        #model_inputs = tokenizer(examples["text"])
        #return {"len": len(model_inputs['input_ids'])}

        model_inputs = tokenizer(examples["text"], max_length=tokenizer_config['max_len'], truncation=True, padding='max_length')
        model_inputs["labels"] = examples['label']
        return model_inputs

    train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=8)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, batch_size=8)

    """import matplotlib.pyplot as plt
    lens = eval_dataset['len']
    lens = pd.DataFrame({'len': lens})
    lens.hist(bins=30)
    plt.show()"""

    print('set up Trainer')

    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        num_train_epochs=5,  # total number of training epochs
        **dataloader_config,
        warmup_ratio=0.1,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        save_strategy='epoch',
        logging_steps=50,
        save_steps=1,
        eval_steps=100,
        evaluation_strategy="steps",  # evaluate each `logging_steps`
        run_name=config['name'],
        no_cuda=False
    )

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset
        compute_metrics=compute_metrics
    )

    trainer.train()
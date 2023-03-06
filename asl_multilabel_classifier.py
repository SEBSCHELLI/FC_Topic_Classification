import os
import wandb
import pandas as pd
import ast
import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainerCallback, TrainerState, \
    TrainerControl
from transformers.models.roberta import RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

print(os.getcwd())
if "FC_Topic_Classification" not in os.getcwd():
    os.chdir("/home/schellsn/FC_Topic_Classification")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class EvaluateCB(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print('compute metrics')

        epoch = state.epoch
        metrics = {'epoch': epoch}

        # train data
        train_pred_output = trainer.predict(train_dataset)

        predictions = (train_pred_output.predictions > 0) * 1
        labels = train_pred_output.label_ids

        train_metrics = {}
        for idx, cat in enumerate(mlb.classes_):
            ap = average_precision_score(labels[:, idx], train_pred_output.predictions[:, idx])
            pre = precision_score(labels[:, idx], predictions[:, idx])
            re = recall_score(labels[:, idx], predictions[:, idx])
            train_metrics.update({f'{cat}_ap': ap,
                                  f'{cat}_prec': pre,
                                  f'{cat}_rec': re,
                                  })

        map = average_precision_score(labels, train_pred_output.predictions)
        train_metrics['map'] = map

        train_metrics = {f'train/{k}': v for k, v in train_metrics.items()}
        metrics.update(train_metrics)

        # dev data
        dev_pred_output = trainer.predict(dev_dataset)

        predictions = (dev_pred_output.predictions > 0) * 1
        labels = dev_pred_output.label_ids

        dev_metrics = {}
        for idx, cat in enumerate(mlb.classes_):
            if cat in dev_cats:
                ap = average_precision_score(labels[:, idx], dev_pred_output.predictions[:, idx])
                pre = precision_score(labels[:, idx], predictions[:, idx])
                re = recall_score(labels[:, idx], predictions[:, idx])
                dev_metrics.update({f'{cat}_ap': ap,
                                    f'{cat}_prec': pre,
                                    f'{cat}_rec': re,
                                    })

            dev_data[f'{cat}_pred'] = predictions[:, idx]
            dev_data[f'{cat}_score'] = sigmoid(dev_pred_output.predictions[:, idx])

        map = average_precision_score(labels, dev_pred_output.predictions)
        dev_metrics['map'] = map

        dev_metrics = {f'dev/{k}': v for k, v in dev_metrics.items()}
        metrics.update(dev_metrics)

        dev_data.to_pickle(f'{output_dir}/dev_data_epoch_{epoch}.pkl', protocol=4)

        # test data
        test_pred_output = trainer.predict(test_dataset)
        predictions = (test_pred_output.predictions > 0) * 1
        labels = test_pred_output.label_ids

        test_metrics = {}
        for idx, cat in enumerate(mlb.classes_):
            ap = average_precision_score(labels[:, idx], test_pred_output.predictions[:, idx])
            pre = precision_score(labels[:, idx], predictions[:, idx])
            re = recall_score(labels[:, idx], predictions[:, idx])
            test_metrics.update({f'{cat}_ap': ap,
                                 f'{cat}_prec': pre,
                                 f'{cat}_rec': re,
                                 })

            test_data[f'{cat}_pred'] = predictions[:, idx]
            test_data[f'{cat}_score'] = sigmoid(test_pred_output.predictions[:, idx])

        map = average_precision_score(labels, test_pred_output.predictions)
        test_metrics['map'] = map

        test_metrics = {f'test/{k}': v for k, v in test_metrics.items()}
        metrics.update(test_metrics)

        test_data.to_pickle(f'{output_dir}/test_data_epoch_{epoch}.pkl', protocol=4)

        wandb.log(metrics)


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASL_RobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=False):
        super().__init__(config)
        self.criterion = AsymmetricLossOptimized(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip, disable_torch_grad_focal_loss=disable_torch_grad_focal_loss)

        wandb.config.gamma_neg = gamma_neg
        wandb.config.gamma_pos = gamma_pos
        wandb.config.clip = clip
        wandb.config.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

        loss = self.criterion(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == '__main__':
    model_id = 'roberta-base'

    input_type = 'claim'

    model_configs = {'roberta-base': {'name': 'roberta-base',
                                      'tokenizer_config': {'pretrained_model_name_or_path': 'roberta-base',
                                                           'max_len': 100 if input_type == 'claim' else 512},
                                      # 100 when claims
                                      'dataloader_config': {
                                          'per_device_train_batch_size': 64 if input_type == 'claim' else 8,
                                          'per_device_eval_batch_size': 512 if input_type == 'claim' else 8}}
                     }

    config = model_configs[model_id]

    dataloader_config = config['dataloader_config']

    tokenizer_config = config['tokenizer_config']

    wandb.init(project="FC_Topic_Clf",
               name="",
               tags=["ASL"]
               )

    wandb.config.update(dataloader_config)
    wandb.config.update(tokenizer_config)
    wandb.config.model_name = config['name']
    wandb.config.input_type = input_type

    output_dir = f'FC_Topic_Clf/{wandb.run.name}'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    claimskg_df_with_tags = pd.read_pickle('claimskg_df_tags.pkl', compression='gzip')
    claim_topics_gold = pd.read_pickle('claim_topics_gold.pkl')

    claimskg_df_with_tags = claimskg_df_with_tags[claimskg_df_with_tags['transformed_extra_tags'].apply(lambda x: True if len(x) > 0 else False)]
    claimskg_df_with_tags = claimskg_df_with_tags[~claimskg_df_with_tags['claimReview_url'].isin(claim_topics_gold['claimReview_url'])]

    test_ws = "fullfact"
    wandb.config.test_ws = test_ws
    train_data = claimskg_df_with_tags[claimskg_df_with_tags['claimReview_source'] != test_ws]
    dev_data = claimskg_df_with_tags[claimskg_df_with_tags['claimReview_source'] == test_ws]
    dev_cats = dev_data.explode("transformed_extra_tags")['transformed_extra_tags'].unique().tolist()
    # train_data = claimskg_df_with_tags.sample(frac=0.8, random_state=0, replace=False).reset_index(drop=True).copy()
    # dev_data = claimskg_df_with_tags[~claimskg_df_with_tags['claimReview_url'].isin(train_data['claimReview_url'])].copy()

    mlb = MultiLabelBinarizer()
    mlb = mlb.fit(claimskg_df_with_tags['transformed_extra_tags'].tolist())

    train_labels = mlb.transform(train_data['transformed_extra_tags'].tolist()) * 1.0
    train_data['label'] = train_labels.tolist()

    dev_labels = mlb.transform(dev_data['transformed_extra_tags'].tolist()) * 1.0
    dev_data['label'] = dev_labels.tolist()

    test_data = claim_topics_gold
    test_labels = mlb.transform(test_data['gold_tags'].tolist()) * 1.0
    test_data['label'] = test_labels.tolist()

    print('load model and tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
    model = ASL_RobertaForSequenceClassification.from_pretrained(config['name'],
                                                                 num_labels=len(mlb.classes_),
                                                                 problem_type='multi_label_classification',
                                                                 gamma_neg=4,
                                                                 gamma_pos=0,
                                                                 clip=0.05,
                                                                 disable_torch_grad_focal_loss=False).cuda()

    #start_from_features = True
    start_from_features = "roberta.encoder.layer.11"
    wandb.config.start_from_features = start_from_features
    if start_from_features:
        if type(start_from_features) == bool:
            until_layer = "classifier"
        elif type(start_from_features) == str:
            until_layer = start_from_features
        req_grad = False
        for n, param in model.named_parameters():
            param.requires_grad = req_grad
            if until_layer in n:
                req_grad = True

            print(f'Parameters {n} require grad: {param.requires_grad}')

    if input_type == "claim":
        train_data["text"] = train_data["claimReview_claimReviewed"].str.lower()
        dev_data["text"] = dev_data["claimReview_claimReviewed"].str.lower()
        test_data["text"] = test_data["claimReview_claimReviewed"].str.lower()

    elif input_type == "claim+text":
        train_data["text"] = train_data["claimReview_claimReviewed"].str.lower() + " " + train_data["extra_body"].str.lower()
        dev_data["text"] = dev_data["claimReview_claimReviewed"].str.lower() + " " + dev_data["extra_body"].str.lower()
        test_data["text"] = test_data["claimReview_claimReviewed"].str.lower() + " " + test_data["extra_body"].str.lower()

    elif input_type == "text":
        train_data["text"] = train_data["extra_body"].str.lower()
        dev_data["text"] = dev_data["extra_body"].str.lower()
        test_data["text"] = test_data["extra_body"].str.lower()

    print('transform data to Datasets')
    train_dataset = Dataset.from_pandas(train_data[['text', 'label']])
    dev_dataset = Dataset.from_pandas(dev_data[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_data[['text', 'label']])

    print('tokenize tweets')


    def tokenize_function(examples):
        model_inputs = tokenizer(examples["text"], max_length=tokenizer_config['max_len'], truncation=True,
                                 padding='max_length')
        model_inputs["labels"] = examples['label']
        return model_inputs


    train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=8)
    dev_dataset = dev_dataset.map(tokenize_function, batched=False, batch_size=8)
    test_dataset = test_dataset.map(tokenize_function, batched=False, batch_size=8)

    """
    import matplotlib.pyplot as plt

    def tokenize_function(examples):
        model_inputs = tokenizer(examples["text"])
        return {"len": len(model_inputs['input_ids'])}

    eval_dataset = eval_dataset.map(tokenize_function, batched=False, batch_size=1)
    lens = eval_dataset['len']
    lens = pd.DataFrame({'len': lens})
    lens.hist(bins=30)
    plt.show()
    """

    print('set up Trainer')

    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        num_train_epochs=10,  # total number of training epochs
        **dataloader_config,
        warmup_ratio=0.1,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        save_strategy='epoch',
        logging_steps=50,
        run_name=config['name'],
        no_cuda=False
    )

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        callbacks=[EvaluateCB]
    )

    trainer.train()

    claimskg_df_with_tags = pd.read_pickle('claimskg_df_tags.pkl', compression='gzip')
    labels = mlb.transform(claimskg_df_with_tags['transformed_extra_tags'].tolist()) * 1.0
    claimskg_df_with_tags['label'] = labels.tolist()

    if input_type == "claim":
        claimskg_df_with_tags["text"] = claimskg_df_with_tags["claimReview_claimReviewed"].str.lower()

    elif input_type == "claim+text":
        claimskg_df_with_tags["text"] = claimskg_df_with_tags["claimReview_claimReviewed"].str.lower() + " " + claimskg_df_with_tags["extra_body"].str.lower()

    elif input_type == "text":
        claimskg_df_with_tags["text"] = claimskg_df_with_tags["extra_body"].str.lower()

    dataset = Dataset.from_pandas(claimskg_df_with_tags[['text', 'label']])
    dataset = dataset.map(tokenize_function, batched=True, batch_size=8)

    pred_output = trainer.predict(dataset)

    predictions = (pred_output.predictions > 0) * 1

    for idx, cat in enumerate(mlb.classes_):
        claimskg_df_with_tags[f'{cat}_pred'] = predictions[:, idx]
        claimskg_df_with_tags[f'{cat}_score'] = sigmoid(pred_output.predictions[:, idx])

    claimskg_df_with_tags.to_pickle(f'{output_dir}/final_data.pkl', protocol=4)

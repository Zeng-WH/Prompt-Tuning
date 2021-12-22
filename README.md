# Prompt-Tuning
Implementation of "The Power of Scale for Parameter-Efficient Prompt Tuning"
Currently, we support the following huggigface models:

- `BartForConditionalGeneration`
## Setup

```
conda create -n prompt-tuning python==3.7.0
pip install -r requirements.txt
```

## Data

The task of summarization supports custom CSV and JSONLINES formats. 

You can use `data2json.py` to transformer data to JSONLINES formats.

#### Custom CSV Files

If it's a csv file the training and validation files should have a column for the inputs texts and a column for the summaries.

If the csv file has just two columns as in the following example:

```
text,summary
"I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder","I'm sitting in a room where I'm waiting for something to happen"
"I see trees so green, red roses too. I see them bloom for me and you. And I think to myself what a wonderful world. I see skies so blue and clouds so white. The bright blessed day, the dark sacred night. And I think to myself what a wonderful world.","I'm a gardener and I'm a big fan of flowers."
"Christmas time is here. Happiness and cheer. Fun for all that children call. Their favorite time of the year. Snowflakes in the air. Carols everywhere. Olden times and ancient rhymes. Of love and dreams to share","It's that time of year again."
```

The first column is assumed to be for `text` and the second is for summary.

If the csv file has multiple columns, you can then specify the names of the columns to use:

```
    --text_column text_column_name \
    --summary_column summary_column_name \
```

For example if the columns were:

```
id,date,text,summary
```

and you wanted to select only `text` and `summary`, then you'd pass these additional arguments:

```
    --text_column text \
    --summary_column summary \
```

#### Custom JSONLINES Files

The second supported format is jsonlines. Here is an example of a jsonlines custom data file.

```
{"text": "I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder", "summary": "I'm sitting in a room where I'm waiting for something to happen"}
{"text": "I see trees so green, red roses too. I see them bloom for me and you. And I think to myself what a wonderful world. I see skies so blue and clouds so white. The bright blessed day, the dark sacred night. And I think to myself what a wonderful world.", "summary": "I'm a gardener and I'm a big fan of flowers."}
{"text": "Christmas time is here. Happiness and cheer. Fun for all that children call. Their favorite time of the year. Snowflakes in the air. Carols everywhere. Olden times and ancient rhymes. Of love and dreams to share", "summary": "It's that time of year again."}
```

Same as with the CSV files, by default the first value will be used as the text record and the second as the summary record. Therefore you can use any key names for the entries, in this example `text` and `summary` were used.

And as with the CSV files, you can specify which values to select from the file, by explicitly specifying the corresponding key names. In our example this again would be:

```
    --text_column text \
    --summary_column summary \
```

## Train

```
bash run_train.sh
```

You can adjust the values for the arguments `--train_file`, `--validation_file` in `run_train.sh`

To control the prompt length, you can adjust the values for the arguments `--pre_seq_len` in `run_train.sh`.

Other setting, such as `learning rate`, `batch_size`, you can also adjust in `run_train.sh`.

## Test

```
bash run_test.sh
```

You can adjust the values for the arguments  `--test_file` in `run_test.sh`

Other setting, you can also adjust in `run_test.sh`. The generated summary is in `output_dir/generated_predictions.txt`

## Citation

```
@misc{lester2021power,
      title={The Power of Scale for Parameter-Efficient Prompt Tuning}, 
      author={Brian Lester and Rami Al-Rfou and Noah Constant},
      year={2021},
      eprint={2104.08691},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

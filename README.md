# Logistic regression based Italian sentiment classifier. #

## Installation, documentation ##

Published results were produced in Python 3 programming environment on Linux Mint 18 Cinnamon operating system. Instructions for installation assume the usage of PyPI package manager.<br/>

Install dependencies if needed: pip install -r requirements.txt

### Instructions: ###

Train the classifier:<br/>

You need a train set in a csv format. The csv file should have columns id, text and target (with sentiment scores). 
To train the classifier just run:

```
python train.py --train_dataset 'pathToTheCSV'
```

Predict sentiment:<br/>

If you want to predict sentiment for just one document/tweet, you can do this in the following way:

```
python predict.py --input_string "some Italian text"
```

If you want to predict sentiment for an entire corpus, call this:

```
python predict.py --input_path example_data/example.tsv --text_column text --output_path results.tsv
```
With argument --input_path you define the path to an input .tsv file (see the example file for details about the format) containing text documents (one per line) that need to be classified. The .tsv file should have column names. Argument --text_column represent the name of the column in the .tsv file that contains document's text. 

The script returns a .tsv file with sentiment predictions for each input document. The path of the output file can be defined with the argument --output_path and defaults to 'results.tsv'.

* [Knowledge Technologies Department](http://kt.ijs.si), Jožef Stefan Institute, Ljubljana



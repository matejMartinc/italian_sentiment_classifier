# Logistic regression based Italian sentiment classifier. #

## Installation, documentation ##

Published results were produced in Python 3 programming environment on Linux Mint 18 Cinnamon operating system. Instructions for installation assume the usage of PyPI package manager.<br/>

Install dependencies if needed: pip install -r requirements.txt

Note that versions of scikit-learn newer than 0.20.3 will not work!

### Instructions: ###

Predict sentiment:<br/>
```
python predict.py --input_path example_data/example.tsv --text_column text --output_path results.tsv
```
With argument --input_path you define the path to an input .tsv file (see the example file for details about the format) containing text documents (one per line) that need to be classified. The .tsv file should have column names. Argument --text_column represent the name of the column in the .tsv file that contains document's text. 

The script returns a .tsv file with sentiment predictions for each input document. The path of the output file can be defined with the argument --output_path and defaults to 'results.tsv'.

* [Knowledge Technologies Department](http://kt.ijs.si), Jožef Stefan Institute, Ljubljana



from sentiment_analysis import preprocess, createFeatures
import os
import argparse
import joblib
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default='model/lr_clf_sentiment_python3_new.pkl', type=str,
                        help="Path to trained model")
    parser.add_argument("--input_string",
                        default="",
                        type=str,
                        help="Path to input tsv file.")
    parser.add_argument("--input_path",
                        default="example_data/example.tsv",
                        type=str,
                        help="Path to input tsv file.")
    parser.add_argument("--text_column",
                        default="text",
                        type=str,
                        help="Name of the column containing text.")
    parser.add_argument("--output_path",
                        default="results.tsv",
                        type=str,
                        help="Path to output file.")
    args = parser.parse_args()


    if len(args.input_string) > 0:
        corpus = [args.input_string]
    else:
        df = pd.read_csv(args.input_path, encoding='utf8', sep='\t')
        corpus = df[args.text_column].tolist()
    df_data = pd.DataFrame({'text': corpus})


    df_prep = preprocess(df_data)
    df_data = createFeatures(df_prep)

    X = df_data
    clf = joblib.load(args.model_path)
    y_pred_gender = clf.predict(X)

    if len(args.input_string) > 0:
        print("Sentiment: ", y_pred_gender[0])
    else:
        df_results = pd.DataFrame({'sentiment': y_pred_gender})
        df_results = pd.concat([df, df_results], axis=1)
        df_results.to_csv(args.output_path, encoding='utf8', sep='\t', index=False)
        print('Done, predictions written to', args.output_path)

# -*- coding: utf-8 -*-
from predict import Predict
from utils import init_logger
import argparse

def predict_func(pred_config):
  predict = Predict()
  print(pred_config.input_file)
  print(pred_config.model_dir)
  lucy_data = predict.preprocess_data(pred_config.input_file)
  model, tok = predict.load_model_n_tokenizer(pred_config.model_dir, len(lucy_data))
  result_set = predict.predict_n_save_result(model, tok, lucy_data,'./output/')
  predict.get_classification_report(result_set)

def main():
  print("====================================Everything is Done. Have a good day :)")


if __name__ == '__main__':
  init_logger()
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_dir", default='./model/best_model', type=str, help="Path to load model")
  parser.add_argument("--input_file", default='./data/predict_lucy.csv', type=str, help="Input file for prediction")
  pred_config = parser.parse_args()
  predict_func(pred_config)
  main()
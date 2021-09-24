import os
import time
import tensorflow as tf
from tokenization_kobert import KoBertTokenizer
import numpy as np
import pandas as pd
from transformers import *
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


class Predict():
    
  def __init__(self,  max_seq_len=128, num_epochs=5, num_batch=16, initializer_range=0.2):
    self.max_seq_len = max_seq_len # seqeunce 최대 길이
    self.num_epochs = num_epochs # 기본 epochs
    self.num_batch = num_batch # 기본 batch 사이즈
    self.initializer_range = initializer_range # 기본 initializer_range
    self.label_dict = {'0': 'IT/과학',
                      '1': '경제',
                      '2': '문화',
                      '3': '미용/건강',
                      '4': '사회',
                      '5': '생활',
                      '6': '스포츠',
                      '7': '연예',
                      '8': '정치'}
    self.LR = 2e-5

  def model_config(self, dataset_length=300):
    warmup_ratio = 0.1
    t_total = dataset_length * self.num_epochs
    warmup_step = int(t_total * warmup_ratio)
    self.LR = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=2e-5, first_decay_steps=warmup_step)
    model = TFBertModel.from_pretrained("monologg/kobert", from_pt=True)
    return model
  
  # model 생성 
  def create_model(self, model):
    """
      tips: class config에 따른 Pretrained Model을 반환합니다.
      Args:
          self : __init__을 통해 설정된 모델 config list
      Returns:
          model : tensorflow TFBertModel
    """

    input_ids_layer = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32)
    attention_masks_layer = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32)
    token_type_ids_layer = tf.keras.layers.Input(shape=(self.max_seq_len,), dtype=tf.int32)

    outputs = model([input_ids_layer, attention_masks_layer, token_type_ids_layer])
    pooled_output = outputs[1]

    optimizer = tf.keras.optimizers.Adam(learning_rate=self.LR)
    
    pooled_output = tf.keras.layers.Dropout(0.5)(pooled_output)
    prediction = tf.keras.layers.Dense(9, activation='softmax', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.initializer_range))(pooled_output)
    cls_model = tf.keras.Model([input_ids_layer, attention_masks_layer, token_type_ids_layer], prediction)
    cls_model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'])
    cls_model.summary()

    return cls_model
  
  def load_model_n_tokenizer(self, model_path, dataset_length):
    """
      tips: Pretrained Model과 Tokenizer를 반환합니다.
      Args:
          model_path : 모델이 저장된 path.
          dataset_length : 데이터셋의 length.
      Returns:
          model : tensorflow TFBertModel
          tok : KoBertTokenizer
    """
    model = self.model_config(dataset_length)
    model = self.create_model(model)
    model.load_weights(model_path)
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    return model, tokenizer

  #전처리 안된 데이터 전처리
  def preprocess_data(self, data_path, data_colname='contents', label_colname='category'):
    """
      tips: csv 데이터를 받아 지정된 column의 내용을 preprocess 합니다.
      Args:
          data_path : csv데이터의 path
          data_colname : 지정할 column명
          label_colname : 지정할 label명
      Returns:
          lucy_data : DataFrame
    """
    lucy_data = pd.read_csv(data_path)

    lucy_data[label_colname+'_tag'] = lucy_data[label_colname] #텍스트 데이터 보존

    encoder = LabelEncoder()
    encoder.fit(lucy_data[label_colname])
    self.label_dict = dict(zip(range(len(encoder.classes_)), encoder.classes_))

    lucy_data[label_colname] = encoder.transform(lucy_data[label_colname])

    lucy_data[data_colname] = lucy_data[data_colname].str.replace("\(.*\)|\s-\s.*"," " ,regex=True)
    lucy_data[data_colname] = lucy_data[data_colname].str.replace("\[.*\]|\s-\s.*"," ",regex=True)
    lucy_data[data_colname] = lucy_data[data_colname].str.replace("\<.*\>|\s-\s.*"," ",regex=True)
    lucy_data[data_colname] = lucy_data[data_colname].str.replace("무단전재 및 재배포 금지"," ",regex=True)
    lucy_data[data_colname] = lucy_data[data_colname].str.replace("무단 전재 및 재배포 금지"," ",regex=True)
    lucy_data[data_colname] = lucy_data[data_colname].str.replace("©"," ",regex=True)
    lucy_data[data_colname] = lucy_data[data_colname].str.replace("ⓒ"," ",regex=True)
    lucy_data[data_colname] = lucy_data[data_colname].str.replace("저작권자"," ",regex=True)
    lucy_data[data_colname] = lucy_data[data_colname].str.replace(".* 기자", " ", regex=True) #기자 이름에서 오는 유사도 차단
    lucy_data[data_colname] = lucy_data[data_colname].str.replace("사진 = .*", " ", regex=True) #사진 첨부 문구 삭제
    lucy_data[data_colname] = lucy_data[data_colname].str.replace("사진=.*", " ", regex=True) #사진 첨부 문구 삭제
    lucy_data[data_colname] = lucy_data[data_colname].str.replace('\"', "",regex=True)
    lucy_data[data_colname] = lucy_data[data_colname].str.replace("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-.]+)", " ", regex=True) #이메일 주소에서 오는 유사도 차단
    lucy_data[data_colname] = lucy_data[data_colname].str.replace("\n"," ")
    lucy_data[data_colname] = lucy_data[data_colname].str.replace("\r"," ")
    lucy_data[data_colname] = lucy_data[data_colname].str.replace("\t"," ")
    lucy_data[data_colname] = lucy_data[data_colname].str.replace( "\’" , "", regex=True)
    # lucy_data[data_colname] = lucy_data[data_colname].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]"," ")
    lucy_data[data_colname] = lucy_data[data_colname].str.replace("[ ]{2,}"," ",regex=True)
    
    return lucy_data

  def sentence_prediction(self, example, tokenizer):
    input_ids, attention_masks, token_type_ids = [], [], []

    input_id = tokenizer.encode(example, max_length=self.max_seq_len, pad_to_max_length=True)
        
    # attention_mask는 실제 단어가 위치하면 1, 패딩의 위치에는 0인 시퀀스.
    padding_count = input_id.count(tokenizer.pad_token_id)
    attention_mask = [1] * (self.max_seq_len - padding_count) + [0] * padding_count
        
    # token_type_id는 세그먼트 임베딩을 위한 것으로 이번 예제는 문장이 1개이므로 전부 0으로 통일.
    token_type_id = [0] * self.max_seq_len

    input_ids.append(input_id)
    attention_masks.append(attention_mask)
    token_type_ids.append(token_type_id)

    input_ids = np.array(input_ids)
    attention_masks = np.array(attention_masks)
    token_type_ids = np.array(token_type_ids)
    return [input_ids, attention_masks, token_type_ids]

  def evaluation_predict(self, sentence, model, tok):
    data_x = self.sentence_prediction(sentence, tok)
    predict = model.predict(data_x)

    # print('예측 결과 수치', predict)
    # print(predict)
    predict_answer = np.argmax(predict[0])
    predict_value = predict[0][predict_answer]
    
    if predict_answer == 0:
      print("(IT/과학 확률 : %.2f) IT/과학 뉴스입니다." % (1-predict_value))
      return 0
    elif predict_answer == 1:
      print("(경제 확률 : %.2f) 경제 뉴스입니다." % predict_value)
      return 1
    elif predict_answer == 2:
      print("(문화 확률 : %.2f) 문화 뉴스입니다." % predict_value)
      return 2
    elif predict_answer == 3:
      print("(미용/건강 확률 : %.2f) 미용/건강 뉴스입니다." % predict_value)
      return 3
    elif predict_answer == 4:
      print("(사회 확률 : %.2f) 사회 뉴스입니다." % predict_value)
      return 4
    elif predict_answer == 5:
      print("(생활 확률 : %.2f) 생활 뉴스입니다." % predict_value)
      return 5
    elif predict_answer == 6:
      print("(스포츠 확률 : %.2f) 스포츠 뉴스입니다." % predict_value)
      return 6
    elif predict_answer == 7:
      print("(연예 확률 : %.2f) 연예 뉴스입니다." % predict_value)
      return 7
    elif predict_answer == 8:
      print("(정치 확률 : %.2f) 정치 뉴스입니다." % predict_value)
      return 8

  #정수를 라벨 텍스트로 디코딩
  def category_decoding(self, lucy_data, label_colname='predict'):
    """
      tips: 정수로 된 라벨을 model에 지정된 텍스트 label로 디코딩.
      Args:
          lucy_data : dataframe 형식의 데이터
          label_colname : 라벨의 컬럼명
      Returns:
          lucy_data : DataFrame
    """
    new_column = label_colname+'_tag'
    lucy_data[new_column] = lucy_data[label_colname].astype(str) # 새로운 컬럼 생성
    lucy_data[new_column] = lucy_data[new_column].astype(str) #String형 컬럼으로 바꾸기

    for key, value in self.label_dict.items():
      lucy_data[new_column] = lucy_data[new_column].str.replace(str(key), value)

    print("===============Data decoding success! ")
    return lucy_data

  def predict_n_save_result(self, model, tok, predict_set, save_directory, encoding_type='utf-8-sig'):
    """
      tips: 예측 및 예측 로그 파일 생성
      Args:
          model : 모델
          tok: tokenizer
          predict_set : 예측할 데이터 셋
          save_directory : 저장할 디렉토리
          encoding_type : csv의 default는 utf-8-sig
      Returns:
          predict_set : DataFrame
    """
    
    #원 데이터 복제, 컬럼생성 및 초기화
    result = predict_set
    result['predict'] = -1

    for idx, x in enumerate(tqdm(predict_set['contents'])):
      out_val = self.evaluation_predict(x, model, tok)
      result['predict'][idx] = out_val

    result = self.category_decoding(result)

    now = int(round(time.time() * 1000))
    filename = 'news_predict_' + str(now) + '.csv'
    save_path = os.path.join(save_directory, filename)
    result.to_csv(save_path, encoding=encoding_type) #한글이면 euc-kr, utf-8-sig 
    print("===============Thank you for waiting. Predicting your dataset Finally Finish! Please Check this directory : ", save_path)
    
    return result

  def get_classification_report(self, predict_set, label_colname="category", predict_colname="predict"):
    """
      tips: 분류 모델 평가 지표 확인
      Args:
          predict_set : 예측된 데이터 셋
          label_colname : 레이블 컬럼명
          predict_colname : 예측된 레이블 컬럼명
      Returns:
          report : classification_report
    """
    label_list = self.label_dict.values()
    report = classification_report(predict_set[label_colname], predict_set[predict_colname], target_names=label_list)
    print("=======================SCORE OF THIS PREDICTION : ")
    print(report)
    return report
 
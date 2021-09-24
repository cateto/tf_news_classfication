

# KoBERT-news-classfication

- KoBERT를 이용한 언론 주제 분류(Document Classfication)
- 현재 모델 미포함
- u2skind@gmail.com



#### Requirements

- Python >= 3.6
- tensorflow == 2.6
- sentencepiece
- transformers == 4.4.2
- tensorflow_addons == 0.14.0



#### How to install

```console
!pip install tensorflow == 2.6
!pip install transformers==4.4.2
!pip install sentencepiece
!pip install tensorflow_addons
```



#### Prediction

```console
python main.py --input_file {INPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}

[for custom setting]

OR

python main.py

[for default setting]
model dir : './model/best_model'
input file : './data/predict_lucy.csv'
```



#### Structure

```console
├── data             		   # data for predict
├── model                      # tensorflow pretrained model
├── output                     # predicted data (csv)
├── main.py                    
├── predict.py
├── README.md
├── tokenization_kobert.py
└── utils.py
```




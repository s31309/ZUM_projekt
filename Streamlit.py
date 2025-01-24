import streamlit as st
import pandas as pd
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import contractions
import spacy
import torch

# MACIERZE POMYŁEK
st.title('Przewidywanie profesji na podstawie opisu')
st.header('Macierze pomyłek dla klasycznego modelu ML i modelu na bazie transformera')

model = st.selectbox('Wybierz model:', ('Klasyczny', 'Transformer'))
conf_matr_classicML = Image.open('D:\ZUM_PROJEKT\Classical_model_confusion_matrix.png')
conf_matr_transformer = Image.open('D:\ZUM_PROJEKT\Transformer_model_confusion_matrix.png') 

if model == 'Klasyczny':
    st.image(conf_matr_classicML)
elif model == 'Transformer':
    st.image(conf_matr_transformer)
else:
    st.warning("Wybierz model")

#RAPORTY KLASYFIKACJI
st.header('Raport klasyfikacji dla wybranego modelu')
model_raport = st.selectbox('Wybierz model:', ('Klasyczny', 'Transformer'), key = 'model_selection')

if model_raport == 'Klasyczny':
    path = 'D:\ZUM_PROJEKT\classification_report_classicalML.txt'
elif model_raport == 'Transformer':
    path = 'D:\ZUM_PROJEKT\classification_report_transformer_model.txt'
else:
    st.warning("Wybierz model")

def load_classification_report(path):
    with open(path, "r") as f:
        lines = f.readlines()

    data = []
    for line in lines[2:-3]:  
        if line.strip():  
            split_line = line.split()
            if len(split_line) == 5:  
                class_name = split_line[0]
                precision, recall, f1_score = map(float, split_line[1:4])  
                support = int(split_line[4])  
                data.append([class_name, precision, recall, f1_score, support]) 

   
    columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    df = pd.DataFrame(data, columns=columns)
    return df

if path:
    df = load_classification_report(path)
    st.write("Raport klasyfikacji:")
    st.dataframe(df)


#KRZYWA ROC
st.header('Krzywa ROC')
ROC_curve = Image.open('D:\ZUM_PROJEKT\ROC_curve.png')
st.image(ROC_curve)


#PREDYKCJA NA PODSTAWIE OPISU
st.header('Model w działaniu')
st.text('Model przewiduje spośród 28 zawodów: accountant, architect, attorney, chiropractor, comedian, composer, dentist, dietitian, dj, filmmaker, interior_designer, journalist, model, nurse, painter, paralegal, pastor, personal_trainer, photographer, physician, poet, professor, psychologist, rapper, software_engineer, surgeon, teacher, yoga_teacher')


model_dir = r'D:\ZUM_PROJEKT\model\transformer_model'

model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

text = st.text_input('Podaj opis profesji (po angielsku):')
text = text.lower()

def expand_abbreviations(text):
  expanded_words = []
  for word in text.split():
    expanded_words.append(contractions.fix(word))
  return ' '.join(expanded_words)

text = expand_abbreviations(text)

nlp = spacy.load("en_core_web_md", disable=['ner', 'parser'])
nlp.add_pipe('sentencizer')

def remove_stopwords(text):
    return ' '.join(filter(lambda x: x not in nlp.Defaults.stop_words, text.split()))

text = remove_stopwords(text)

def lemmatize(text):
  return ' '.join([x.lemma_ for x in nlp(text)])

text = lemmatize(text)

label_mapping = {
    0: "accountant",
    1: "architect",
    2: "attorney",
    3: "chiropractor",
    4: "comedian",
    5: "composer",
    6: "dentist",
    7: "dietitian",
    8: "dj",
    9: "filmmaker",
    10: "interior_designer",
    11: "journalist",
    12: "model",
    13: "nurse",
    14: "painter",
    15: "paralegal",
    16: "pastor",
    17: "personal_trainer",
    18: "photographer",
    19: "physician",
    20: "poet",
    21: "professor",
    22: "psychologist",
    23: "rapper",
    24: "software_engineer",
    25: "surgeon",
    26: "teacher",
    27: "yoga_teacher"
}

inputs = tokenizer(
    text, 
    return_tensors="pt",
    padding='max_length',
    truncation=True,
    max_length=128 
)

model.eval() 
with torch.no_grad(): 
    outputs = model(**inputs)

logits = outputs.logits
predicted_classes = torch.argmax(logits, dim=-1)
predicted_labels = [label_mapping[label.item()] for label in predicted_classes]

if text:
    st.write("Predykcja modelu:", predicted_labels)
else:
    st.warning('Podaj opis profesji')
# -*- coding: utf-8 -*-
"""
https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
@author: Puneet Gupta
"""

import Loader
import Text_Preprocess
from pickle import load
from keras.models import load_model
from Model_handling import extract_features
from Model_handling import generate_desc
from shutil import copyfile
import os

from nltk.translate.bleu_score import corpus_bleu
from PIL import Image

dataset_root_dir = '/var/www/html/TextBasedImageSearch/Dataset/'
code_root_dir = '/var/www/html/TextBasedImageSearch/'
weights = code_root_dir + "ResNet50/model_19.h5"
model = load_model(weights)
# load the tokenizer
tokenizer = load(open(code_root_dir + 'tokenizer_resnet50.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34

# load and prepare the photograph
photo = extract_features(code_root_dir+'upload/file.jpeg')
# generate description
predicted_description = generate_desc(model, tokenizer, photo, max_length)
print_description = ' '.join(predicted_description.split(' ')[1:-1])

desc_file = open(code_root_dir+'upload/description.txt',"w")
desc_file.write(print_description)
desc_file.close()

testFile = '/var/www/html/TextBasedImageSearch/Dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
testImagesLabel = Loader.load_set(testFile)
test_descriptions = Loader.load_clean_descriptions(dataset_root_dir + 'descriptions.txt', testImagesLabel)
matchedFiles = set()

for img in testImagesLabel:
    if len(matchedFiles) > 50:
        break
    actual, predicted = list(), list()
    yhat = predicted_description.split()
    predicted.append(yhat)
    references = [d.split() for d in test_descriptions[img]]
    actual.append(references) 
    bleu_score_1 = corpus_bleu(actual, predicted, weights=(1, 0, 0, 0))
    bleu_score_2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu_score_3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.34, 0))
    bleu_score_4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    bleu_score = ( 8*bleu_score_4 + 4*bleu_score_3 + 2*bleu_score_2 + bleu_score_1 )/15
    if bleu_score > 0.3:
        matchedFiles.add(img)
        continue

print(len(matchedFiles))

path = '/var/www/html/TextBasedImageSearch/Dataset/Flickr8k_Dataset/Flicker8k_Dataset/'

matched_img_file = open('/var/www/html/TextBasedImageSearch/upload/matched_images.txt', "w")

folder = '/var/www/html/TextBasedImageSearch/upload/matched-images'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        # elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

desc_text = Text_Preprocess.load_text(dataset_root_dir + 'Flickr8k_text/Flickr8k.token.txt')
descriptions = Text_Preprocess.load_description(desc_text)

i = 0
for img in matchedFiles:
    img_path = path + img + '.jpg'
    i += 1
    matched_img_file.write(descriptions[img][0] + '\n')
    copyfile(img_path, folder + '/' + format(i, '03d') + '.jpg')

matched_img_file.close()

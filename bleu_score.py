from nltk.translate.bleu_score import corpus_bleu
from global_variable import dataset_dir
import Loader


def find_matched_image(
        image_caption_list,
        text_description
):
    """
    :param image_caption_list:
    :param text_description:
    :return:
    """
    matchedimages_file = set()
    test_descriptions = Loader.load_clean_descriptions(
        dataset_dir + 'descriptions.txt',
        image_caption_list)
    for image in image_caption_list:
        actual, predicted = list(), list()
        yhat = text_description.split()
        predicted.append(yhat)
        references = [d.split() for d in test_descriptions[image]]
        actual.append(references)
        bleu_score_1 = corpus_bleu(actual, predicted, weights=(1, 0, 0, 0))
        bleu_score_2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
        bleu_score_3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.34, 0))
        bleu_score_4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_score = (4 * bleu_score_4 + 3 * bleu_score_3 + 2 * bleu_score_2 + bleu_score_1) / 10
        if bleu_score > 0.1:
            matchedimages_file.add(image)
            continue

    return matchedimages_file
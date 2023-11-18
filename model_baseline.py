def split_wordlist(wordlist):
    masculine_words = []
    feminine_words = []

    for item in wordlist:
        masculine_word = item['word_masculine']
        feminine_word = item['word_feminine']
        
        masculine_words.append(masculine_word)
        feminine_words.append(feminine_word)

    return masculine_words, feminine_words

def split_testset(testset, range=0.05):
    texts = []
    binary_scores = []
    binary_labels = []

    for item in testset:
        text = item['text']
        binary_label = item['binary_label']
        binary_score = item['binary_score'] if binary_label else 1 - item['binary_score']
        binary_label = binary_label if abs(binary_score-0.5) > range else 2

        texts.append(text)
        binary_labels.append(binary_label)
        binary_scores.append(binary_score)

    return texts, binary_labels, binary_scores


def classify_gender(utterance, masculine_words, feminine_words):
    # Convert the utterance to lower case and split into words
    words = utterance.lower().split()

    # Count the occurrences of masculine and feminine words
    masculine_count = sum(word in masculine_words for word in words)
    feminine_count = sum(word in feminine_words for word in words)

    score = masculine_count / (masculine_count + feminine_count)

    # Classify the utterance based on the counts
    if masculine_count > feminine_count:
        return 1, score # male
    elif feminine_count > masculine_count:
        return 0, score # female
    else:
        return 2, score # gender-neutral

def predict_gender(texts, masculine_words, feminine_words):
    predict_labels = []
    predict_scores = []

    for text in texts:
        predict_label, predict_score = classify_gender(text, masculine_words, feminine_words)

        predict_labels.append(predict_label)
        predict_scores.append(predict_score)

    return predict_labels, predict_scores



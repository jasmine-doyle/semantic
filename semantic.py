import spacy
nlp = spacy.load('en_core_web_md')


# =================== 1st code extract from document ======================

# this code tests and displays the similarity between three words
print("\n1st Code Extract:\n")
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(f"{word1} & {word2} : {word1.similarity(word2)}")
print(f"{word3} & {word2} : {word3.similarity(word2)}")
print(f"{word3} & {word1} : {word3.similarity(word1)}")

# cat and monkey have the highest similarity as they are both animals
# then monkey and banana have the next highest as monkeys eat bananas
# lastly cat and banana have the least in common


# my own examples
word4 = nlp("car")
word5 = nlp("plane")
word6 = nlp("sky")
print(f"{word4} & {word5} : {word4.similarity(word5)}")
print(f"{word6} & {word5} : {word6.similarity(word5)}")
print(f"{word6} & {word4} : {word6.similarity(word4)}")

# car and plane have the highest similarity as they are both modes of transport
# then plane and sky have the next highest as planes fly in the sky
# lastly car and sky have the least in common



# =================== 2nd code extract from document ======================

# this code tests and displays the similarity between four words
print("\n2nd Code Extract:\n")
tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(f"{token1.text} & {token2.text} : {token1.similarity(token2)}")



# =================== 3rd code extract from document ======================

# this code tests and displays the similarity between sentences
print("\n3rd Code Extract:\n")
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


# =================== 3rd code extract from document ======================

# when using the simpler language model 'en_core_web_sm' the following warning is displayed:

# UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be based on the tagger, parser and NER, which may not give useful similarity judgements. 
# This may happen if you're using one of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors.
# You can always add your own word vectors, or use one of the larger models instead if available.

# all of the values for the similarities change, and some even change the order of which words are more similar to each other
# it is much better to use 'en_core_web_md' for this
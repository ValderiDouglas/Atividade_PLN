import nltk
import numpy as np
import pandas as pd
import copy as cp

texto = """
But, soft! what light through yonder window breaks? It is the east, and Juliet is the sun. Arise, fair sun, and kill the envious moon, Who is already sick and pale with grief, That thou her maid art far more fair than she: Be not her maid, since she is envious; Her vestal livery is but sick and green And none but fools do wear it; cast it off. It is my lady, O, it is my love! O, that she knew she were! She speaks yet she says nothing: what of that? Her eye discourses; I will answer it. I am too bold, 'tis not to me she speaks: Two of the fairest stars in all the heaven, Having some business, do entreat her eyes To twinkle in their spheres till they return. What if her eyes were there, they in her head? The brightness of her cheek would shame those stars, As daylight doth a lamp; her eyes in heaven Would through the airy region stream so bright That birds would sing and think it were not night. See, how she leans her cheek upon her hand! O, that I were a glove upon that hand, That I might touch that cheek! Ay me! She speaks: O, speak again, bright angel! for thou art As glorious to this night, being o'er my head As is a winged messenger of heaven Unto the white-upturned wondering eyes Of mortals that fall back to gaze on him When he bestrides the lazy-pacing clouds And sails upon the bosom of the air. O Romeo, Romeo! wherefore art thou Romeo? Deny thy father and refuse thy name; Or, if thou wilt not, be but sworn my love, And I'll no longer be a Capulet.
"""

# Tokenização por sentenças com NLTK
sentencas = nltk.sent_tokenize(texto)
print("Tokenização por sentenças:")
for i, sentenca in enumerate(sentencas, 1):
    print(f"{i}. {sentenca}")

# Tokenização por palavras
palavras = nltk.word_tokenize(texto)
print("\nBag of Words :")
bag_of_words = cp.deepcopy(palavras)
np.random.shuffle(bag_of_words)
print(bag_of_words[:20])  

# POS tagging com NLTK
print("\nUsing natural language toolkit:")
pos_tags = nltk.pos_tag(palavras)
print("POS tags:")
print(pos_tags[:10])
pos_tags_df = pd.DataFrame(pos_tags, columns=['Word', 'POS']).T

print("A tokenização por sentenças divide o texto em unidades completas de pensamento,")
print("tokenização por palavras separa cada palavra individualmente, fazendo análise detalhada")

pos_tags_set_nltk = set(tag for _, tag in pos_tags)
print("\nPOS tags NLTK :", pos_tags_set_nltk)
from tokenization_srna import SrnaTokenizer


text = "<|cap|>"
srnatok = SrnaTokenizer()
text = srnatok.prepare_for_tokenization(text)

print(text)
import requests
import string

url = "https://raw.githubusercontent.com/spyguessgame-boop/own_dataset/refs/heads/main/data.txt"

response = requests.get(url)
response.raise_for_status()

text_data = response.text

text_data = text_data[:1000]

#removing punctuations

for p in string.punctuation:
    text_data=text_data.replace(p,"")

#tokenization

tokens=text_data.split()

print("Tokens: ")
print(tokens)

#number of tokens

print("Total tokens = ",len(tokens))

#frequency

freq={}

for word in tokens:
    if word in freq:
        freq[word]+=1
    else:
        freq[word] = 1

most_frequent=max(freq, key=freq.get)

print("Most frequent token : ",most_frequent)
print("Count : ",freq[most_frequent])


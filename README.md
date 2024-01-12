# ML
Collection of ML projects I am using for learning.

***LyricGen***
The LyricGen folder is a folder that uses NLP (Natural Language Processing) to predict lyrics based off of a prompt. The *.txt* files contain the lyrics and sources.
These packages are used if you wanted to test the program:
```
pip install tensorflow numpy
```
*Of course if you're lazy like me then I will just explain my code instead*

All of this was done in a google colab python notebook cause it takes way too long to compile (40 epochs, large dataset with ngrams, running with 4GB ram = not fun)

Splitting the data into a list and removing duplicates. Tested multiple times and choruses ended up having a negative effect on the output
![image](https://github.com/burtonjong/ML/assets/108391733/4b069bd9-e299-4a3d-a9ba-dc22ca8b60cf)

Each word is giving an index with the tokenizer
![image](https://github.com/burtonjong/ML/assets/108391733/6361f87a-aa75-4965-9af0-39d7fc5b2231)

Ngrams are created with the for loop and then padded to match the max sentence length (or else there would be a problem when building the model due to shape differences)
![image](https://github.com/burtonjong/ML/assets/108391733/6731e4d6-f14c-459a-8f95-5e7792eaa52e)
We are using x and y as the training set in entirety as the user is the one that gives the "test set" (or prompt)
Put in simple terms, the x is every index leading up to the label y. For example:

If we were given the sentence *Hi I am Drake* and with the tokenizer
Word | Hi | I | Am | Drake 
--- | --- | --- | --- |--- 
Index | 4 | 6 | 3 | 9

Assuming this is the max sentence length,
*x* would look something like:
```
[
[4,6]
[4,6,3]
[4,6,3,9] # In this example, 4,6,3 is the input x and 9 is y, or the label.
]
```
While *y* would look something like if 9 is the label:
```
[[0 ,0, 0, 0, 0, 0, 0, 0, 1]...]
```
Basically, lets say you were given the sentence *Hi I Am*. You would assume the 4th word is Drake. What if you were given *Hi I*? Well, you could also assume that the 3rd word is Am. Thus the model is trained in to predict the next word with the actual next word as the label. Wow!!!!

![image](https://github.com/burtonjong/ML/assets/108391733/3394767a-5d87-42bf-8e67-df2739699f43)
The model is then trained and fit. I don't think I am qualified to actually explain what each layer does but I will give some pretty short explanations:

**Embedding**: Given the bunch of words, it uses vectors to find how they relate to eachother and then also outputs another vector for each word

**LSTM**: LSTM or Long Short-Term Memory is basically just a way for the model to understand and remember the lryics. Bidirectional means it reads it from left to right.

**Dense**: Determines outputs based on how many words we have with a softmax activation. 

![image](https://github.com/burtonjong/ML/assets/108391733/ce7870b9-a619-40c3-aba0-da9ba163cf8c)
![image](https://github.com/burtonjong/ML/assets/108391733/4c820dd6-d449-4797-a308-80e86a9944b5)

Well Drake swears a lot. Not much I can do








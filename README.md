# AI Recipe Generator

## Problem statement:

AI Recipe Generator takes any ingredients as input, and by using different models and techniques it will generate a recipe based on the number of ingredients you provide.
It's a simple way to get personalised recipes tailored to what you have in your kitchen.

## Description of Dataset

Source for scraping:

1) Beautiful Soup

2) Selenium

## Preprocessing Techniques Applied

1) Data cleaning

2) Feature Selection

3) Normalization

4) Tokenization   

## Which ML Models Selected and Why?

TfidfVectorizer for feature extraction and a Multinomial Naive Bayes classifier, which is commonly used for text classification tasks.
The TfidfVectorizer converts the text data into numerical features, and the Multinomial Naive Bayes model is then trained on these features.

## Which DL Models Selected and Why? 

Simple neural network with an embedding layer for text representation.
Using an embedding layer to convert the ingredients into dense vectors, flattens the vectors, and then passes them through a couple of dense layers. The model is trained using the softmax activation function for multiclass classification.


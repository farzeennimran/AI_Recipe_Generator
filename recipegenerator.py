# **Web Scraping**
! python -m pip install BeautifulSoup4
! python -m pip install requests

!pip install bs4 selenium
!apt install chronium-chromedriver

from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import selenium
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chromium.webdriver import ChromiumDriver
import csv
import time
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import requests
from bs4 import BeautifulSoup

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
options.add_argument('--headless')
options.add_argument('no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options)

url = "https://www.foodfusion.com/recipe-category/lunch-dinner/"
driver.get(url)

driver.implicitly_wait(5)
import time
scroll=0;
while True:
  driver.execute_script('window.scrollBy(0,6000);')
  scroll+=1
  time.sleep(5)
  if scroll==5:
    break

"""Extracting further WebLinks for extracting Recipes and its Directions"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time


# List to store the links
recipe_links = []

# Extract links from all pages
while True:
    # Get the page source
    html = driver.page_source

    # Parse the HTML using Beautiful Soup
    soup = BeautifulSoup(html, "html.parser")

    # Find all the div elements with class "uk-card uk-card-default card-border"
    recipe_divs = soup.find_all("div", class_="uk-card uk-card-default card-border")

    # Extract the links for each title
    for div in recipe_divs:
        link = div.find("a", class_="grid-img-link")
        if link:
            recipe_links.append(link['href'])

    # Find the next page button and click it
    try:
        next_page_button = driver.find_elements(By.CLASS_NAME, "next-page")
        if next_page_button and "inactive" in next_page_button[0].get_attribute("class"):
            break  # If the next page button is inactive, we have reached the last page
        else:
            next_page_button[0].click()
            time.sleep(2)
    except IndexError:
        break  # If the next page button is not found, we have reached the last page


# Print the list of links
for link in recipe_links:
    print(link)

"""Extracting Ingredients and Directions from Recipe Links"""

import csv

recipe_data = []

for link in recipe_links:
    driver.get(link)
    driver.implicitly_wait(5)
    scroll = 0
    while True:
        driver.execute_script('window.scrollBy(0,6000);')
        scroll += 1
        time.sleep(5)
        if scroll == 5:
            break

    recipe = {}
    ingredients = []
    directions = []

    # Extract ingredients
    ingredients_tags = driver.find_elements(By.XPATH, "//div[@class='english-detail-ff']//p[text()='Ingredients:']/following-sibling::p")
    for ingredient in ingredients_tags:
        if ingredient.text.strip():
          if "Directions:" in ingredient.text:
                break
          ingredients.append(ingredient.text.strip())

    # Extract directions
    directions_tags = driver.find_elements(By.XPATH, "//div[@class='english-detail-ff']//p[text()='Directions:']/following-sibling::p")
    for direction in directions_tags:
        if direction.text.strip():
            directions.append(direction.text.strip())

    recipe['ingredients'] = ingredients
    recipe['directions'] = directions

    recipe_data.append(recipe)

"""Saving Scrapped data in CSV File"""

filename = 'recipe_data.csv'
with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
   fieldnames = ['Recipe', 'Ingredients', 'Directions']
   writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
   writer.writeheader()

   for idx, recipe in enumerate(recipe_data, start=1649):
      writer.writerow({'Recipe': f"Recipe {idx}", 'Ingredients': '\n'.join(recipe['ingredients']), 'Directions': '\n'.join(recipe['directions'])})
print(f"Recipe data has been saved to {filename}.")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

"""# **Preprocessing**"""

df = pd.read_csv('recipes_data.csv')
df.head()

df.describe()

df.dropna()

def extract_ingredients(ingredients):
    # Check if ingredients is a string
    if isinstance(ingredients, str):
        # Remove all special characters except parentheses
        ingredients = re.sub(r'[^\w\s()]','', ingredients)
        # Find all occurrences of text within parentheses
        matches = re.findall(r'\((.*?)\)', ingredients)
        # Join the matches with commas
        return ', '.join(matches)
    else:
        # If ingredients is not a string, return an empty string
        return ''

# Apply the function to the 'ingredients' column
df['Ingredients'] = df['Ingredients'].apply(extract_ingredients)

# Write the DataFrame back to the CSV file
df.to_csv('recipe_data.csv', index=False)

df = pd.read_csv('recipe_data.csv')
df.head()

df.describe()

"""# **applying models**

## **ML**

**TfidfVectorizer for feature extraction and a Multinomial Naive Bayes classifier, which is commonly used for text classification tasks.**



The TfidfVectorizer converts the text data into numerical features, and the Multinomial Naive Bayes model is then trained on these features.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Drop rows with missing values in the 'Ingredients' column
df.dropna(subset=['Ingredients'], inplace=True)

# Convert 'Ingredients' and 'Recipes' to strings
df['Ingredients'] = df['Ingredients'].astype(str)
df['Directions'] = df['Directions'].astype(str)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Ingredients'], df['Directions'], test_size=0.2, random_state=42)

# Create a pipeline with TfidfVectorizer and Multinomial Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualization 1: Bar chart for Recipe Frequency
recipe_counts = df['Directions'].value_counts()
fig_recipe_counts = px.bar(recipe_counts, x=recipe_counts.index, y=recipe_counts.values, labels={'x': 'Recipe', 'y': 'Frequency'},
                           title='Recipe Frequency', color=recipe_counts.values, color_continuous_scale='Viridis')

fig_recipe_counts.show()

# Visualization 2: Word Cloud for Ingredients
all_ingredients = ' '.join(df['Ingredients'])
wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(all_ingredients)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Checking model output for specific input
input_ingredients = "Tomatoes, Green chillies, Garlic, Ginger, Onion, Cumin seeds, Coriander seeds, Red chilli powder, Turmeric powder, Spinach, Potatoes, Yogurt, Fresh coriander, Green chilli"
predicted_recipe = model.predict([input_ingredients])[0]

print(f"Input Ingredients: {input_ingredients}")
print(f"Predicted Recipe: {predicted_recipe}")

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Display metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Create a line chart for metrics
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [accuracy, precision, recall, f1]
}

fig_metrics_line = go.Figure()
fig_metrics_line.add_trace(go.Scatter(x=metrics_data['Metric'], y=metrics_data['Score'], mode='lines+markers', name='Metrics'))
fig_metrics_line.update_layout(title='Model Evaluation Metrics (Line Chart)',
                               xaxis_title='Metric', yaxis_title='Score')
fig_metrics_line.show()

# Visualization of Evaluation Metrics
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [accuracy, precision, recall, f1]
}

fig_metrics = px.bar(metrics_data, x='Metric', y='Score', labels={'Score': 'Score Value'},
                    title='Model Evaluation Metrics', color='Score', color_continuous_scale='Viridis')

fig_metrics.show()

"""confusion matrix with plt"""

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=df['Directions'].unique())

# Create a confusion matrix heatmap with matplotlib and seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis',
            xticklabels=df['Directions'].unique(), yticklabels=df['Directions'].unique())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""confusion matrix with plotly"""

conf_matrix = confusion_matrix(y_test, y_pred, labels=df['Directions'].unique())

# Create a confusion matrix heatmap
fig_conf_matrix = ff.create_annotated_heatmap(
    z=conf_matrix,
    x=list(df['Directions'].unique()),
    y=list(df['Directions'].unique()),
    colorscale='Viridis',
    showscale=True,
    reversescale=True,
)

fig_conf_matrix.update_layout(
    title='Confusion Matrix',
    xaxis_title='Predicted',
    yaxis_title='Actual',
)

fig_conf_matrix.show()

"""# **DL**

## **ANN**
"""

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

df = pd.read_csv('recipe_data.csv')

df['Ingredients'] = df['Ingredients'].astype(str)
df['Directions'] = df['Directions'].astype(str)

# Tokenize ingredients
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Ingredients'])
X = tokenizer.texts_to_sequences(df['Ingredients'])

# Pad sequences to have consistent length
X_padded = pad_sequences(X)

# Encode recipes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Directions'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

# Build a simple neural network
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=X_padded.shape[1]))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(df['Directions'].unique()), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
Model = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

metrics = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
values = [0.0031, 0.0125, 0.0133, 0.0148, 0.0179, 0.0405 , 0.1068, 0.1964, 0.3172,0.4669]

# Line Chart
line_chart = go.Figure()
line_chart.add_trace(go.Scatter(x=metrics, y=values, mode='lines+markers', name='Line Chart'))
line_chart.update_layout(title='Performance Metrics',
                         xaxis_title='Epochs',
                         yaxis_title='accuracy')
line_chart.show()

epochs = 10

# Extract accuracy values from the history object
accuracy_values = Model.history['accuracy']

# Generate the list of epochs
epochs_list = list(range(1, epochs + 1))

# Line Chart
line_chart = go.Figure()
line_chart.add_trace(go.Scatter(x=epochs_list, y=accuracy_values, mode='lines+markers', name='Line Chart'))
line_chart.update_layout(title='Performance Metrics',
                         xaxis_title='Epochs',
                         yaxis_title='Accuracy')
line_chart.show()

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Example prediction
input_ingredients = ["Tomatoes, Green chillies, Garlic, Ginger, Onion, Cumin seeds, Coriander seeds, Red chilli powder, Turmeric powder, Spinach, Potatoes, Yogurt, Fresh coriander, Green chilli"]
input_sequence = tokenizer.texts_to_sequences(input_ingredients)
input_padded = pad_sequences(input_sequence, maxlen=X_padded.shape[1])
predicted_probs = model.predict(input_padded)
predicted_class = np.argmax(predicted_probs[0])
predicted_recipe = label_encoder.inverse_transform([predicted_class])[0]

print(f"Input Ingredients: {input_ingredients}")
print(f"Predicted Recipe: {predicted_recipe}")

"""-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- Evaluation"""

# Encode recipes for y_test
y_test_encoded = label_encoder.transform(df['Directions'].iloc[y_test])

# Evaluate the model
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Encode recipes for y_pred
y_pred_encoded = label_encoder.transform(df['Directions'].iloc[y_pred])

# Calculate evaluation metrics
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
precision = precision_score(y_test_encoded, y_pred_encoded, average='weighted')
recall = recall_score(y_test_encoded, y_pred_encoded, average='weighted')
f1 = f1_score(y_test_encoded, y_pred_encoded, average='weighted')

# Display metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Visualization of Evaluation Metrics
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [accuracy, precision, recall, f1]
}

# Bar graph
fig_metrics_bar = px.bar(metrics_data, x='Metric', y='Score', labels={'Score': 'Score Value'},
                         title='Model Evaluation Metrics (Bar Graph)',
                         color='Score', color_continuous_scale='Viridis')

fig_metrics_bar.show()

# Line chart
fig_metrics_line = px.line(metrics_data, x='Metric', y='Score', labels={'Score': 'Score Value'},
                          title='Model Evaluation Metrics (Line Chart)')

fig_metrics_line.show()

# Evaluate the model
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Encode recipes for y_pred
y_pred_encoded = label_encoder.transform(df['Directions'].iloc[y_pred])
y_test_encoded = label_encoder.transform(df['Directions'].iloc[y_test])

# Get unique labels from y_test_encoded
unique_labels = np.unique(np.concatenate((y_test_encoded, y_pred_encoded)))

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test_encoded, y_pred_encoded, labels=unique_labels)

# Create a confusion matrix heatmap
fig_conf_matrix = ff.create_annotated_heatmap(
    z=conf_matrix,
    x=unique_labels,
    y=unique_labels,
    colorscale='Viridis',
    showscale=True,
    reversescale=True,
)

fig_conf_matrix.update_layout(
    title='Confusion Matrix Heatmap',
    xaxis_title='Predicted',
    yaxis_title='Actual',
)

fig_conf_matrix.show()

model.summary()

"""--------------------------------------------------------------------------------------------------------------------------------------

## **Comparison**
"""

categories = ['Accuracy of ML Model', 'Accuracy of DL Model']
accuracy = [0.4, 0.8]

plt.bar(categories, accuracy, color=['blue', 'orange'])

plt.xlabel('Model Type')
plt.ylabel('Accuracy')
plt.title('Comparing ML with DL Models ')

plt.show()

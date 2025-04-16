# intellihack_raccoons_task2

This repository contains a Python-based intent classification system that leverages Natural Language Processing (NLP) techniques to identify user intents from text inputs. The implementation includes training a Support Vector Machine (SVM) model on an expanded dataset and provides a command-line interface for testing intent classification.

## Features

- **Intent Detection**: Determines the user's intent with a confidence score.
- **Dataset Expansion**: Automatically triples the provided dataset to enhance training.
- **Customizable Threshold**: Allows setting a confidence threshold for intent prediction.
- **Real-Time User Interaction**: Accepts user input for on-the-fly intent classification.

## Requirements

- Python 3.x
- Required libraries (install via `pip`):
  - `scikit-learn`
  - `numpy`

## How It Works

1. **Dataset Preparation**: A predefined dataset of intents and examples is expanded to improve model performance.
2. **Model Training**: A pipeline consisting of a TF-IDF vectorizer and an SVM classifier is trained on the dataset.
3. **Intent Classification**: The model predicts the intent of user inputs based on the highest confidence score.

## Usage

### 1. Clone the repository:

   ```bash
   git clone https://github.com/Walapalam/intellihack_raccoons_task2.git
   cd intellihack_raccoons_task2
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```
### 3. Run the script:

```bash
python q2.py
```

### 4. Enter your message at the prompt to see the predicted intent and confidence score. Type quit to exit the program.

## Example Interaction
```bash
Enter your message (or type 'quit' to exit): Hello
Predicted Intent: Greet, Confidence: 0.85

Enter your message (or type 'quit' to exit): What's the weather like today?
Predicted Intent: Inquiry, Confidence: 0.76
```

## File Overview
 - q2.py: Main script containing the dataset, model training, and classification logic.

## Future Improvements
- Include additional intents and examples for better coverage.
- Enhance the fallback mechanism for handling low-confidence predictions.
- Provide a web or GUI-based interface for easier user interaction.

# Transformer from Scratch

This repository demonstrates how to implement a Transformer model from scratch and apply it to a simple question-answering task. The project includes the processes of model pretraining, fine-tuning, and inference.

## Project Structure

- `transformer.py`: Core implementation of the Transformer model
- `vocab.py`: Vocabulary construction and tokenization
- `pretrain.py`: Model pretraining script
- `train.py`: Model fine-tuning script
- `demo.py`: Interactive demo script
- `util.py`: Utility functions

## Main Features

1. **Transformer Model Implementation**: Includes core components such as multi-head attention, positional encoding, encoder, and decoder layers.

2. **Pretraining**: Autoregressive language model pretraining using Wikipedia data.

3. **Fine-tuning**: Fine-tuning the pretrained model on a specific task (birthplace prediction).

4. **Vocabulary Handling**: Tokenization and vocabulary management using spaCy.

5. **Interactive Demo**: Allows users to interact with the trained model in real-time.

## Usage

1. **Environment Setup**:
   ```
   pip install torch spacy tqdm
   python -m spacy download en_core_web_sm
   ```

2. **Data Preparation**:
   Place Wikipedia data in `data/wiki.txt`, and task-specific data in the `data/` directory.

    Here is a brief description of each dataset file:

    1. **birth_dev.tsv**: 
    - This file contains data used for model development and validation. Each entry includes an entity pair and their relationship (e.g., a person's name and birthplace).

    2. **birth_places_train.tsv**: 
    - This is the training dataset, typically used to train the model. It contains numerous questions and answers to help the model learn how to extract relationships from text.

    3. **birth_test_inputs.tsv**: 
    - This file contains test input data for evaluating the model after training. It includes only questions, with the aim for the model to predict the relationships.

    4. **wiki.txt**: 
    - This file contains raw text data extracted from sources like Wikipedia. It serves as pretraining data for the language model.
3. **Vocabulary Construction**:
   ```
   python vocab.py
   ```

4. **Pretraining**:
   ```
   python pretrain.py
   ```

5. **Fine-tuning**:
   ```
   python train.py
   ```

6. **Interactive Demo**:
   ```
   python demo.py
   ```

## Model Architecture

The core implementation of the Transformer model is in the `transformer.py` file.
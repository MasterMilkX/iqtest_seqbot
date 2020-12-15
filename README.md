# MegaCat
#### Sequencer bot for the IIJCAI-2020 MAIQ Competition
Developed by Megan Charity and Catalina Jaramillo

---
### Dependencies
`pip -r requirements.txt`
### Execution
1. Verify that the `classifier/` folder exists
2. Verify that the `data/` folder exists and contains the `seq-public.json`, `seq-public.answer.json`, and the `seq-private.json` files inside
3. Run `python iq_bot.py`
---
### Main source code
`iq_bot.py` - predicts on the training data set (*seq-public.json*) and evaluates accuracy (*seq-public.answer.json*) and predicts answers for the test dataset (*seq-private.json*). This file contains the model creators [*hybrid*, *index*, *recursive*] for the sequence predictions. The program returns two reports:
   1. `seq_train_accuracy.txt`- an accuracy report describing the questions evaluated, the questions correctly answered, and an accuracy percentage. 
   2. `seq_train_response.txt` - a question response file with the answers of the bot and the true answers corresponding to the question ID number in the training dataset.
   
`stats.py` - returns accuracy stats used in the paper for the training dataset. Statistics include following information: accuracy by model, accuracy by question type, bad data parsed percentage, and average error from true answer
**classifier/** - location of the pre-trained classifier model. Can be retrained by running the notebooks [sequence classifier.ipynb, sequence classifier 3 classes.ipynb]


---
### Notebooks
These notebooks are for preliminary testing for the different components of the algorithm:

`hint_classes.ipynb` - parses the sequence hints in the training dataset into semi-readable format

`seq_parser.ipynb` - cleans and parses the sequences into machine readable format (removes any sequences with invalid characters or unparsable format (i.e. square roots, words))

`sequence classifier.ipynb` - builds the classifier model for sorting sequences based on their hint functions. Sorts into 4 different categories, recursive-based patterns, index-based patterns, a hybrid of both recursive and index-based patterns, and an unknown category. The variation notebook *sequence classifier 3 classes* does the same function without the recursive-bases pattern category (since 1 sample was only found in our initial results)

`test_rnn.ipynb` - preliminary experiments for developing the RNN models for sequence predictions. Includes tests for model hyperparameters such as activation functions, layer sizes, learning rates, and epoch numbers

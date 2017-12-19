import warnings
from asl_data import SinglesData
import numpy as np

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # TODO implement the recognizer
    # return probabilities, guesses
    
    # so we receive the test model then apply the test set and return the probabilities given model:word combo.
    
    # take word list - predict using built model - store word and output probability
    
    probabilities = []
    guesses = []
    
    obs_seq = list(test_set.get_all_Xlengths().values())
    
    for test_x, test_xlen in obs_seq:
        probability = {}
        
        for word,model in models.items():
            
            try: 
                score = model.score(test_x,test_xlen)
                probability[word] = score
            except:
                probability[word] = -np.inf
            
        probabilities.append(probability)
        #probabilities is now a word:prob pair
        guess = max([(k,v) for k,v in probability.items()])[0]
        guesses.append(guess)
        
    return (probabilities, guesses)
    

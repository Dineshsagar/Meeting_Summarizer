from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
import nltk
nltk.download('punkt')

def summarize_text(text, num_sentences=5, summarizer_type='lsa'):
    """
    Summarizes the given text using the specified summarizer.
    
    Args:
        text (str): The text to be summarized.
        num_sentences (int): The number of sentences to include in the summary.
        summarizer_type (str): The type of summarizer to use ('lsa', 'text_rank', or 'luhn').
    
    Returns:
        str: The summarized text.
    """
    # Create a PlaintextParser object from the input text
    parser = PlaintextParser.from_string(text, Tokenizer('english'))
    
    # Choose the summarizer based on the specified type
    if summarizer_type == 'lsa':
        summarizer = LsaSummarizer()
    elif summarizer_type == 'text_rank':
        summarizer = TextRankSummarizer()
    elif summarizer_type == 'luhn':
        summarizer = LuhnSummarizer()
    else:
        raise ValueError("Invalid summarizer type. Choose 'lsa', 'text_rank', or 'luhn'.")
    
    # Generate the summary
    summary = [sentence for sentence in summarizer(parser.document, num_sentences)]
    
    # Return the summary as a string
    return ' '.join(str(sentence) for sentence in summary)

# Example usage
text = """ 
text here
"""

summary = summarize_text(text, num_sentences=1, summarizer_type='lsa')
print(summary)


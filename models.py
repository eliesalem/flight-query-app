
print('hello')

import sys
from collections import defaultdict

# Check if the script was called with a query argument
if len(sys.argv) > 1:
    query = sys.argv[1]  # The first command-line argument is the query
    #print(f"Received query: {query}")  # Just print the query for testing
else:
    print("No query provided.")


# Please do not change this cell because some hidden tests might depend on it.
import subprocess
import os
# Initialize Otter
import otter
grader = otter.Notebook()

import copy
import datetime
import math
import os
import random
import re
import sys
import warnings

import wget
import nltk
import sqlite3
import csv
import torch
import torch.nn as nn
import datasets

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers import Regex
from tokenizers.pre_tokenizers import WhitespaceSplit, Split
from tokenizers.processors import TemplateProcessing
from tokenizers import normalizers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast

from cryptography.fernet import Fernet
from func_timeout import func_set_timeout
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from tqdm.notebook import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration

print("hi")

# Set random seeds for reproducibility
SEED = 1234

def reseed(seed=SEED):
    torch.manual_seed(seed)
    random.seed(seed)

reseed()
# Set timeout for executing SQL
TIMEOUT = 3 # seconds

# GPU check: Set runtime type to use GPU where available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print (device)

# Prepare to download needed data
def download_if_needed(source, dest, filename, add_to_path=True):
    os.makedirs(dest, exist_ok=True)  # ensure destination
    if add_to_path:
        sys.path.insert(1, dest)  # add local to path
    if os.path.exists(f"./{dest}{filename}"):
        #print(f"Skipping {filename}")
        cat = 1
    else:
        #print(f"Downloading {filename} from {source}")
        wget.download(source + filename, out=dest)
        #print("", flush=True)


source_path = "https://raw.githubusercontent.com/" "nlp-course/data/master/ATIS/"
code_path = "https://raw.githubusercontent.com/" "nlp-course/data/master/scripts/trees/"
data_path = "data/"


# Grammar to augment for this segment
if not os.path.isfile("data/grammar"):
    download_if_needed(source_path, data_path, "grammar_distrib4.crypt")

    # Decrypt the grammar file
    key = b"bfksTY2BJ5VKKK9xZb1PDDLaGkdu7KCDFYfVePSEfGY="
    fernet = Fernet(key)
    with open(data_path + "grammar_distrib4.crypt", "rb") as f:
        restored = Fernet(key).decrypt(f.read())
    with open(data_path + "grammar", "wb") as f:
        f.write(restored)


# Acquire the datasets (training, development, and test splits of the
# ATIS queries and corresponding SQL queries) and ATIS SQL database
for filename in [
    "atis_sqlite.db",  # ATIS SQL database
    "train_flightid.nl",
    "train_flightid.sql",
    "dev_flightid.nl",
    "dev_flightid.sql",
    "test_flightid.nl",
    "test_flightid.sql",
]:
    download_if_needed(source_path, data_path, filename)

# Download scripts and ATIS database
for filename in ["transform.py"]:  # tree transformations
    download_if_needed(code_path, data_path, filename)

# Import downloaded scripts for parsing augmented grammars
import transform as xform

arithmetic_grammar, arithmetic_augmentations = xform.parse_augmented_grammar(
    """
    ## Sample grammar for arithmetic expressions

    S -> NUM                              : lambda Num: Num
       | S OP S                           : lambda S1, Op, S2: Op(S1, S2)

    OP -> ADD                             : lambda Op: Op
        | SUB
        | MULT
        | DIV

    NUM -> 'zero'                         : lambda: 0
         | 'one'                          : lambda: 1
         | 'two'                          : lambda: 2
         | 'three'                        : lambda: 3
         | 'four'                         : lambda: 4
         | 'five'                         : lambda: 5
         | 'six'                          : lambda: 6
         | 'seven'                        : lambda: 7
         | 'eight'                        : lambda: 8
         | 'nine'                         : lambda: 9
         | 'ten'                          : lambda: 10

    ADD -> 'plus' | 'added' 'to'          : lambda: lambda x, y: x + y
    SUB -> 'minus'                        : lambda: lambda x, y: x - y
    MULT -> 'times' | 'multiplied' 'by'   : lambda: lambda x, y: x * y
    DIV -> 'divided' 'by'                 : lambda: lambda x, y: x / y
    """
)
def interpret(tree, augmentations):
    syntactic_rule = tree.productions()[0]
    #print(f"Syntactic rule: {syntactic_rule}")  # Debugging line
    semantic_rule = augmentations[syntactic_rule]
    child_meanings = [interpret(child, augmentations)
                      for child in tree
                      if isinstance(child, nltk.Tree)]
    #print(f"Child meanings: {child_meanings}")  # Debugging line
    result = semantic_rule(*child_meanings)
    #print(f"Interpretation result: {result}")  # Debugging line
    return result


def parse_and_interpret(string, grammar, augmentations):
  parser = nltk.parse.BottomUpChartParser(grammar)
  parses = parser.parse(string.split())
  for parse in parses:
    parse.pretty_print()
    print("==>", interpret(parse, augmentations))

def constant(value):
  """Return `value`, ignoring any arguments"""
  return lambda *args: value

def first(*args):
  """Return the value of the first (and perhaps only) subconstituent,
     ignoring any others"""
  return args[0]

def numeric_template(rhs):
  """Ignore the subphrase meanings and lookup the first right-hand-side symbol
     as a number"""
  return constant({'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5,
          'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10}[rhs[0]])

def forward(F, A):
  """Forward application: Return the application of the first
     argument to the second"""
  return F(A)

def backward(A, F):
  """Backward application: Return the application of the second
     argument to the first"""
  return F(A)

def second(*args):
  """Return the value of the second subconstituent, ignoring any others"""
  return args[1]

def ignore(*args):
  """Return `None`, ignoring everything about the constituent. (Good as a
     placeholder until a better augmentation can be devised.)"""
  return None

# Process data into CSV files
for split in ['train', 'dev', 'test']:
    src_in_file = f'{data_path}{split}_flightid.nl'
    tgt_in_file = f'{data_path}{split}_flightid.sql'
    out_file = f'{data_path}{split}.csv'

    with open(src_in_file, 'r') as f_src_in, open(tgt_in_file, 'r') as f_tgt_in:
        with open(out_file, 'w') as f_out:
            src, tgt= [], []
            writer = csv.writer(f_out)
            writer.writerow(('src','tgt'))
            for src_line, tgt_line in zip(f_src_in, f_tgt_in):
                writer.writerow((src_line.strip(), tgt_line.strip()))

## NLTK Tokenizer
tokenizer_pattern = '\d+|st\.|[\w-]+|\$[\d\.]+|\S+'
nltk_tokenizer = nltk.tokenize.RegexpTokenizer(tokenizer_pattern)
def tokenize_nltk(string):
  return nltk_tokenizer.tokenize(string.lower())

## Demonstrating the tokenizer
## Note especially the handling of `"11pm"` and hyphenated words.
#print(tokenize_nltk("Are there any first-class flights from St. Louis at 11pm for less than $3.50?"))

dataset = load_dataset(
    "csv",
    data_files={
        "train": f"{data_path}train.csv",
        "val": f"{data_path}dev.csv",
        "test": f"{data_path}test.csv",
    },
)
dataset

train_data = dataset['train']
val_data = dataset['val']
test_data = dataset['test']

MIN_FREQ = 3
unk_token = '[UNK]'
pad_token = '[PAD]'
bos_token = '<bos>'
eos_token = '<eos>'

## source tokenizer
src_tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
src_tokenizer.normalizer = normalizers.Lowercase()
src_tokenizer.pre_tokenizer = Split(Regex(tokenizer_pattern),
                                    behavior='removed',
                                    invert=True)

src_trainer = WordLevelTrainer(min_frequency=MIN_FREQ,
                               special_tokens=[pad_token, unk_token])
src_tokenizer.train_from_iterator(train_data['src'],
                                  trainer=src_trainer)

## target tokenizer
tgt_tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
tgt_tokenizer.pre_tokenizer = WhitespaceSplit()

tgt_trainer = WordLevelTrainer(min_frequency=MIN_FREQ,
                               special_tokens=[pad_token, unk_token,
                                               bos_token, eos_token])
tgt_tokenizer.train_from_iterator(train_data['tgt'],
                                  trainer=tgt_trainer)
tgt_tokenizer.post_processor = \
    TemplateProcessing(single=f"{bos_token} $A {eos_token}",
                       special_tokens=[(bos_token,
                                        tgt_tokenizer.token_to_id(bos_token)),
                                       (eos_token,
                                        tgt_tokenizer.token_to_id(eos_token))])

hf_src_tokenizer = PreTrainedTokenizerFast(tokenizer_object=src_tokenizer,
                                           pad_token=pad_token)
hf_tgt_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tgt_tokenizer,
                                           pad_token=pad_token)

def encode(example):
    example['src_ids'] = hf_src_tokenizer(example['src']).input_ids
    example['tgt_ids'] = hf_tgt_tokenizer(example['tgt']).input_ids
    return example

train_data = train_data.map(encode)
val_data = val_data.map(encode)
test_data = test_data.map(encode)

# Compute size of vocabulary
src_vocab = src_tokenizer.get_vocab()
tgt_vocab = tgt_tokenizer.get_vocab()
'''
print(f"Size of English vocab: {len(src_vocab)}")
print(f"Size of SQL vocab: {len(tgt_vocab)}")
print(f"Index for src padding: {src_vocab[pad_token]}")
print(f"Index for tgt padding: {tgt_vocab[pad_token]}")
print(f"Index for start of sequence token: {tgt_vocab[bos_token]}")
print(f"Index for end of sequence token: {tgt_vocab[eos_token]}")
'''

BATCH_SIZE = 16     # batch size for training and validation
TEST_BATCH_SIZE = 1 # batch size for test; we use 1 to make implementation easier

# Defines how to batch a list of examples together
def collate_fn(examples):
    batch = {}
    bsz = len(examples)
    src_ids, tgt_ids = [], []
    for example in examples:
        src_ids.append(example['src_ids'])
        tgt_ids.append(example['tgt_ids'])

    src_len = torch.LongTensor([len(word_ids) for word_ids in src_ids]).to(device)
    src_max_length = max(src_len)
    tgt_max_length = max([len(word_ids) for word_ids in tgt_ids])

    src_batch = torch.zeros(bsz, src_max_length).long().fill_(src_vocab[pad_token]).to(device)
    tgt_batch = torch.zeros(bsz, tgt_max_length).long().fill_(tgt_vocab[pad_token]).to(device)
    for b in range(bsz):
        src_batch[b][:len(src_ids[b])] = torch.LongTensor(src_ids[b]).to(device)
        tgt_batch[b][:len(tgt_ids[b])] = torch.LongTensor(tgt_ids[b]).to(device)

    batch['src_lengths'] = src_len
    batch['src_ids'] = src_batch
    batch['tgt_ids'] = tgt_batch
    return batch

train_iter = torch.utils.data.DataLoader(train_data,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         collate_fn=collate_fn)
val_iter = torch.utils.data.DataLoader(val_data,
                                       batch_size=BATCH_SIZE,
                                       shuffle=False,
                                       collate_fn=collate_fn)
test_iter = torch.utils.data.DataLoader(test_data,
                                        batch_size=TEST_BATCH_SIZE,
                                        shuffle=False,
                                        collate_fn=collate_fn)

@func_set_timeout(TIMEOUT)
def execute_sql(sql):
  conn = sqlite3.connect('data/atis_sqlite.db')  # establish the DB based on the downloaded data
  c = conn.cursor()                              # build a "cursor"
  c.execute(sql)
  results = list(c.fetchall())
  c.close()
  conn.close()
  return results

def upper(term):
  return '"' + term.upper() + '"'

def weekday(day):
  return f"flight.flight_days IN (SELECT days.days_code FROM days WHERE days.day_name = '{day.upper()}')"

def month_name(month):
  return {'JANUARY' : 1,
          'FEBRUARY' : 2,
          'MARCH' : 3,
          'APRIL' : 4,
          'MAY' : 5,
          'JUNE' : 6,
          'JULY' : 7,
          'AUGUST' : 8,
          'SEPTEMBER' : 9,
          'OCTOBER' : 10,
          'NOVEMBER' : 11,
          'DECEMBER' : 12}[month.upper()]

def airports_from_airport_name(airport_name):
  return f"(SELECT airport.airport_code FROM airport WHERE airport.airport_name = {upper(airport_name)})"

def airports_from_city(city):
  return f"""
    (SELECT airport_service.airport_code FROM airport_service WHERE airport_service.city_code IN
      (SELECT city.city_code FROM city WHERE city.city_name = {upper(city)}))
  """

def null_condition(*args, **kwargs):
  return 1

def depart_around(time):
  return f"""
    flight.departure_time >= {add_delta(miltime(time), -15).strftime('%H%M')}
    AND flight.departure_time <= {add_delta(miltime(time), 15).strftime('%H%M')}
    """.strip()

def add_delta(tme, delta):
    # transform to a full datetime first
    return (datetime.datetime.combine(datetime.date.today(), tme) +
            datetime.timedelta(minutes=delta)).time()

def miltime(minutes):
  return datetime.time(hour=int(minutes/100), minute=(minutes % 100))

atis_grammar, atis_augmentations = xform.read_augmented_grammar('data/grammar', globals=globals())
atis_parser = nltk.parse.BottomUpChartParser(atis_grammar)

def parse_tree(sentence):
  """Parse a sentence and return the parse tree, or None if failure."""
  try:
    parses = list(atis_parser.parse(tokenize_nltk(sentence)))
    if len(parses) == 0:
      return None
    else:
      return parses[0]
  except:
    return None

def check_coverage(sentence_file):
    """ Tests the coverage of the grammar on the sentences in the
    file `sentence_file`. Returns the number of sentences that parse
    as well as the total number of sentences in the file
    """
    parsed = 0
    with open(sentence_file) as sentences:
        sentences = sentences.readlines()[:]
    for sentence in tqdm(sentences):
        if parse_tree(sentence):
            parsed += 1
        else:
            next
    return parsed, len(sentences)

def verify(predicted_sql, gold_sql, silent=True):
    """
    Compare the correctness of the `predicted_sql` by executing on the
    ATIS database and comparing the returned results to the`gold_sql`.
    Arguments:
        predicted_sql: the predicted SQL query
        gold_sql: the reference SQL query to compare against
        silent: print outputs or not
    Returns:
        True if the returned results are the same, otherwise False
    """
    # Execute predicted SQL
    try:
        predicted_result = execute_sql(predicted_sql)
    except BaseException as e:
        if not silent:
            print(f"predicted sql exec failed: {e}")
        return False
    if not silent:
        print("Predicted DB result:\n\n", predicted_result[:10], "\n")

    # Execute gold SQL
    try:
        gold_result = execute_sql(gold_sql)
    except BaseException as e:
        if not silent:
            print(f"gold sql exec failed: {e}")
        return False
    if not silent:
        print("Gold DB result:\n\n", gold_result[:10], "\n")

    # Verify correctness
    if gold_result == predicted_result:
        return True

def rule_based_trial(sentence, gold_sql):
    print("Sentence: ", sentence, "\n")
    tree = parse_tree(sentence)
    print("Parse:\n\n")
    tree.pretty_print()

    predicted_sql = interpret(tree, atis_augmentations)
    print("Predicted SQL:\n\n", predicted_sql, "\n")

    if verify(predicted_sql, gold_sql, silent=False):
        print("Correct!")
    else:
        print("Incorrect!")

# Run this cell to reload augmentations after you make changes to `data/grammar`
atis_grammar, atis_augmentations = xform.read_augmented_grammar(f"{data_path}grammar", globals=globals())
atis_parser = nltk.parse.BottomUpChartParser(atis_grammar)

class Beam:
    """
    Helper class for storing a hypothesis, its score and its decoder hidden state.
    """

    def __init__(self, decoder_state, tokens, score):
        self.decoder_state = decoder_state
        self.tokens = tokens
        self.score = score


class BeamSearcher:
    """
    Main class for beam search.
    """

    def __init__(self, model):
        self.model = model
        self.bos_id = model.bos_id
        self.eos_id = model.eos_id
        self.padding_id_src = model.padding_id_src
        self.V = model.V_tgt

    def beam_search(self, src, src_lengths, K, max_T=15):
        """
        Performs beam search decoding.
        Arguments:
            src: src batch of size (1, max_src_len)
            src_lengths: src lengths of size (1)
            K: beam size
            max_T: max possible target length considered
        Returns:
            a list of token ids and a list of attentions
        """
        finished = []
        all_attns = []

        self.model.eval()

        # Initialize the beam
        memory_bank = self.model.forward_encoder(src, src_lengths)[0]
        encoder_final_state = self.model.forward_encoder(src, src_lengths)[1]
        init_beam = Beam(decoder_state=encoder_final_state, tokens=[torch.tensor(self.bos_id)], score=0)
        beams = [init_beam]

        with torch.no_grad():
            for t in range(max_T):  # main body of search over time steps

                # Expand each beam by all possible tokens y_{t+1}
                all_total_scores = []
                for beam in beams:
                    y_1_to_t, score, decoder_state = beam.tokens, beam.score, beam.decoder_state,
                    
                    y_t = y_1_to_t[-1]
                    src_mask = src.ne(self.padding_id_src)
                    logits, decoder_state, attn = self.model.forward_decoder_incrementally(decoder_state, y_t, memory_bank, src_mask, normalize=True)

                    total_scores = logits + score
                    all_total_scores.append(total_scores)
                    all_attns.append(attn)  # keep attentions for visualization
                    beam.decoder_state = decoder_state  
                all_total_scores = torch.stack(all_total_scores)  # (K, V) when t>0, (1, V) when t=0

                # Find K best next beams
                # The code below has the same functionality as lines 6-12, but is more efficient
                all_scores_flattened = all_total_scores.view(-1)  # K*V when t>0, 1*V when t=0
                topk_scores, topk_ids = all_scores_flattened.topk(K, 0)
                beam_ids = topk_ids.div(self.V, rounding_mode="floor")
                next_tokens = topk_ids - beam_ids * self.V
                new_beams = []
                for k in range(K):
                    beam_id = beam_ids[k]  # which beam it comes from
                    y_t_plus_1 = next_tokens[k]  # which y_{t+1}
                    score = topk_scores[k]
                    beam = beams[beam_id]
                    decoder_state = beam.decoder_state
                    y_1_to_t = beam.tokens
                    new_beam = Beam(decoder_state, y_1_to_t + [y_t_plus_1], score)
                    new_beams.append(new_beam)
                beams = new_beams

                # Set aside completed beams
                for beam in beams:
                    y_t_plus_1 = beam.tokens[-1]
                if y_t_plus_1 == self.eos_id:
                    finished.append(beam)
                    beams.remove(beam)

                # Break the loop if everything is completed
                if len(beams) == 0:
                    break

        # Return the best hypothesis
        if len(finished) > 0:
            finished = sorted(finished, key=lambda beam: -beam.score)
            return [token.item() for token in finished[0].tokens], all_attns
        else:  # when nothing is finished, return an unfinished hypothesis
            return [token.item() for token in beams[0].tokens], all_attns
        
def attention(batched_Q, batched_K, batched_V, mask=None):
    """
    Performs the attention operation and returns the attention matrix
    `batched_A` and the context matrix `batched_C` using queries
    `batched_Q`, keys `batched_K`, and values `batched_V`.

    Arguments:
      batched_Q: (bsz, q_len, D)
      batched_K: (bsz, k_len, D)
      batched_V: (bsz, k_len, D)
      mask: (bsz, q_len, k_len). An optional boolean mask *disallowing*
            attentions where the mask value is *`False`*.
    Returns:
      batched_A: the normalized attention scores (bsz, q_len, k_len)
      batched_C: a tensor of size (bsz, q_len, D).
    """
    # Check sizes
    D = batched_Q.size(-1)
    bsz = batched_Q.size(0)
    q_len = batched_Q.size(1)
    k_len = batched_K.size(1)
    assert batched_K.size(-1) == D and batched_V.size(-1) == D
    assert batched_K.size(0) == bsz and batched_V.size(0) == bsz
    assert batched_V.size(1) == k_len
    if mask is not None:
        assert mask.size() == torch.Size([bsz, q_len, k_len])

    A = torch.bmm(batched_Q, batched_K.transpose(1,2))

    if mask is not None:
        A = A.masked_fill(mask==False, -float('inf'))

    batched_A = torch.softmax(A, dim=-1)
    batched_C = torch.bmm(batched_A, batched_V)

    # Verify that things sum up to one properly.
    assert torch.all(torch.isclose(batched_A.sum(-1),
                                 torch.ones(bsz, q_len).to(device)))
    return batched_A, batched_C

# TODO - implement the `AttnEncoderDecoder` class.
class AttnEncoderDecoder(nn.Module):
    def __init__(self, hf_src_tokenizer, hf_tgt_tokenizer, hidden_size=64, layers=3):
        super().__init__()
        self.hf_src_tokenizer = hf_src_tokenizer
        self.hf_tgt_tokenizer = hf_tgt_tokenizer

        # Keep the vocabulary sizes available
        self.V_src = len(self.hf_src_tokenizer)
        self.V_tgt = len(self.hf_tgt_tokenizer)

        # Get special word ids
        self.padding_id_src = self.hf_src_tokenizer.pad_token_id
        self.padding_id_tgt = self.hf_tgt_tokenizer.pad_token_id
        self.bos_id = self.hf_tgt_tokenizer.get_vocab()[bos_token]
        self.eos_id = self.hf_tgt_tokenizer.get_vocab()[eos_token]

        # Keep hyperparameters available
        self.embedding_size = hidden_size
        self.hidden_size = hidden_size
        self.layers = layers

        # Create essential modules
        self.word_embeddings_src = nn.Embedding(self.V_src, self.embedding_size)
        self.word_embeddings_tgt = nn.Embedding(self.V_tgt, self.embedding_size)

        # RNN cells
        self.encoder_rnn = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=hidden_size // 2,
            num_layers=layers,
            batch_first=True,
            bidirectional=True
        )
        self.decoder_rnn = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=False
        )

        # Final projection layer
        self.hidden2output = nn.Linear(2 * hidden_size, self.V_tgt)

        # Create loss function
        self.loss_function = nn.CrossEntropyLoss(reduction='sum',
                                                 ignore_index=self.padding_id_tgt)

    def forward_encoder(self, src, src_lengths):
        src_embedded = self.word_embeddings_src(src)
        packed = pack(src_embedded, lengths=src_lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, (h, c) = self.encoder_rnn(packed)
        batch_size = len(src_lengths)
        h = h.reshape(self.layers, 2, batch_size, self.hidden_size // 2)
        h = h.transpose(1, 2)
        h = h.reshape(self.layers, batch_size, -1)

    
        c = c.reshape(self.layers, 2, batch_size, self.hidden_size // 2)
        c = c.transpose(1, 2)
        c = c.reshape(self.layers, batch_size, -1)
        final_state = (h, c)

        memory_bank = unpack(packed_outputs, batch_first=True)[0]
        context = None
        return memory_bank, (final_state, context)

    def forward_decoder(self, encoder_final_state, tgt_in, memory_bank, src_mask):
        max_tgt_length = tgt_in.size(1)
        decoder_states = encoder_final_state
        all_logits = []
        for i in range(max_tgt_length):
            logits, decoder_states, _ = self.forward_decoder_incrementally(
                decoder_states,
                tgt_in[:, i],
                memory_bank,
                src_mask,
                normalize=False
            )
            all_logits.append(logits)
        all_logits = torch.stack(all_logits, 1)  # bsz, tgt_len, vocab_tgt
        return all_logits


    def forward(self, src, src_lengths, tgt_in):
        src_mask = src.ne(self.padding_id_src)  # bsz, max_src_len

        # Forward encoder
        memory_bank, encoder_final_state = self.forward_encoder(src, src_lengths)

        # Forward decoder
        logits = self.forward_decoder(encoder_final_state, tgt_in, memory_bank, src_mask)

        return logits

    def forward_decoder_incrementally(self, prev_decoder_states, tgt_in_onestep,
                                      memory_bank, src_mask, normalize=True):
        prev_decoder_state, prev_context = prev_decoder_states
        batch_size = prev_decoder_state[0].size(1)
        tgt_embedding = self.word_embeddings_tgt(tgt_in_onestep.reshape(batch_size, 1))

        if prev_context is not None:
            tgt_embedding += prev_context

        decoder_output, decoder_state = self.decoder_rnn(tgt_embedding, prev_decoder_state)
        mask = src_mask.unsqueeze(1)

        attn_weights, context_vector = attention(decoder_output, memory_bank, memory_bank, mask)

        combined_input = torch.cat((decoder_output, context_vector), dim=-1)
        logits = self.hidden2output(combined_input)

        if normalize:
            logits = torch.log_softmax(logits, dim=-1)

        updated_decoder_states = (decoder_state, context_vector)

        return logits, updated_decoder_states, attn_weights

    def evaluate_ppl(self, iterator):
        self.eval()
        total_loss = 0
        total_words = 0
        for batch in iterator:
            src = batch['src_ids']
            src_lengths = batch['src_lengths']
            tgt_in = batch['tgt_ids'][:, :-1]
            tgt_out = batch['tgt_ids'][:, 1:]
            logits = self.forward(src, src_lengths, tgt_in)
            loss = self.loss_function(logits.reshape(-1, self.V_tgt), tgt_out.reshape(-1))
            total_loss += loss.item()
            total_words += tgt_out.ne(self.padding_id_tgt).float().sum().item()
        return math.exp(total_loss / total_words)

    def train_all(self, train_iter, val_iter, epochs=10, learning_rate=0.001):
        self.train()
        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        best_validation_ppl = float('inf')
        best_model = None
        for epoch in range(epochs):
            total_words = 0
            total_loss = 0.0
            for batch in tqdm(train_iter):
                self.zero_grad()
                tgt = batch['tgt_ids']
                src = batch['src_ids']
                src_lengths = batch['src_lengths']
                tgt_in = tgt[:, :-1].contiguous()
                tgt_out = tgt[:, 1:].contiguous()
                bsz = tgt.size(0)
                logits = self.forward(src, src_lengths, tgt_in)
                loss = self.loss_function(logits.view(-1, self.V_tgt), tgt_out.view(-1))
                num_tgt_words = tgt_out.ne(self.padding_id_tgt).float().sum().item()
                total_words += num_tgt_words
                total_loss += loss.item()
                loss.div(bsz).backward()
                optim.step()

            validation_ppl = self.evaluate_ppl(val_iter)
            self.train()
            if validation_ppl < best_validation_ppl:
                best_validation_ppl = validation_ppl
                self.best_model = copy.deepcopy(self.state_dict())
            epoch_loss = total_loss / total_words
            print(f'Epoch: {epoch} Training Perplexity: {math.exp(epoch_loss):.4f} '
                  f'Validation Perplexity: {validation_ppl:.4f}')

    def predict(self, tokens, K=1, max_T=15):
        beam_searcher = BeamSearcher(self)
        
        src = torch.tensor([hf_src_tokenizer.encode(tokens)])
        src_lengths = torch.tensor([len(src[0])])

        prediction, _ = beam_searcher.beam_search(src, src_lengths, K, max_T)
        prediction = self.hf_tgt_tokenizer.decode(prediction, skip_special_tokens=True)

        return prediction

EPOCHS = 1
# we start with 1 epoch, but for good performance
# you'll want to use many more epochs, perhaps 10
LEARNING_RATE = 1e-4

# Instantiate and train classifier
model = AttnEncoderDecoder(
    hf_src_tokenizer,
    hf_tgt_tokenizer,
    hidden_size=2,
    layers=1,
).to(device)

model.train_all(train_iter, val_iter, epochs=EPOCHS, learning_rate=LEARNING_RATE)

# Evaluate model performance on the validation set using the best model found. 
# The expected value should be < 1.2
model.load_state_dict(model.best_model)
print (f'Validation perplexity: {model.evaluate_ppl(tqdm(val_iter)):.3f}')



# TODO - implement the `AttnEncoderDecoder2` class.
class AttnEncoderDecoder2(nn.Module):
    def __init__(self, hf_src_tokenizer, hf_tgt_tokenizer, hidden_size=64, layers=3):
        super().__init__()
        self.hf_src_tokenizer = hf_src_tokenizer
        self.hf_tgt_tokenizer = hf_tgt_tokenizer

        # Keep the vocabulary sizes available
        self.V_src = len(self.hf_src_tokenizer)
        self.V_tgt = len(self.hf_tgt_tokenizer)

        # Get special word ids
        self.padding_id_src = self.hf_src_tokenizer.pad_token_id
        self.padding_id_tgt = self.hf_tgt_tokenizer.pad_token_id
        self.bos_id = self.hf_tgt_tokenizer.get_vocab()[bos_token]
        self.eos_id = self.hf_tgt_tokenizer.get_vocab()[eos_token]

        # Keep hyperparameters available
        self.embedding_size = hidden_size
        self.hidden_size = hidden_size
        self.layers = layers

        # Create essential modules
        self.word_embeddings_src = nn.Embedding(self.V_src, self.embedding_size)
        self.word_embeddings_tgt = nn.Embedding(self.V_tgt, self.embedding_size)

        # RNN cells
        self.encoder_rnn = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=hidden_size // 2,
            num_layers=layers,
            batch_first=True,
            bidirectional=True
        )
        self.decoder_rnn = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=False
        )

        # Final projection layer
        self.hidden2output = nn.Linear(2 * hidden_size, self.V_tgt)

        # Create loss function
        self.loss_function = nn.CrossEntropyLoss(reduction='sum',
                                                 ignore_index=self.padding_id_tgt)

    def forward_encoder(self, src, src_lengths):
        src_embedded = self.word_embeddings_src(src)
        packed = pack(src_embedded, lengths=src_lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, (h, c) = self.encoder_rnn(packed)
        batch_size = len(src_lengths)
        h = h.reshape(self.layers, 2, batch_size, self.hidden_size // 2)
        h = h.transpose(1, 2)
        h = h.reshape(self.layers, batch_size, -1)

    
        c = c.reshape(self.layers, 2, batch_size, self.hidden_size // 2)
        c = c.transpose(1, 2)
        c = c.reshape(self.layers, batch_size, -1)
        final_state = (h, c)

        memory_bank = unpack(packed_outputs, batch_first=True)[0]
        context = None
        return memory_bank, (final_state, context, None)

    def forward_decoder(self, encoder_final_state, tgt_in, memory_bank, src_mask):
        max_tgt_length = tgt_in.size(1)
        decoder_states = encoder_final_state
        all_logits = []
        for i in range(max_tgt_length):
            logits, decoder_states, _ = self.forward_decoder_incrementally(
                decoder_states,
                tgt_in[:, i],
                memory_bank,
                src_mask,
                normalize=False
            )
            all_logits.append(logits)
        all_logits = torch.stack(all_logits, 1)  # bsz, tgt_len, vocab_tgt
        return all_logits


    def forward(self, src, src_lengths, tgt_in):
        src_mask = src.ne(self.padding_id_src)  # bsz, max_src_len

        # Forward encoder
        memory_bank, encoder_final_state = self.forward_encoder(src, src_lengths)

        # Forward decoder
        logits = self.forward_decoder(encoder_final_state, tgt_in, memory_bank, src_mask)

        return logits

    def forward_decoder_incrementally(self, prev_decoder_states, tgt_in_onestep,
                                      memory_bank, src_mask, normalize=True):
        
        prev_decoder_state, prev_context, prev_output = prev_decoder_states
        batch_size = prev_decoder_state[0].size(1)
        tgt_embedding = self.word_embeddings_tgt(tgt_in_onestep.reshape(batch_size, 1))

        if prev_context is not None:
            tgt_embedding += prev_context

        decoder_output, decoder_state = self.decoder_rnn(tgt_embedding, prev_decoder_state)

        if prev_context is not None:
            _, self_context = attention(decoder_output, prev_output, prev_output)
            prev_output = torch.cat((prev_output, decoder_output), dim = 1)
            torch.add(decoder_output, self_context)
        else:
            prev_output = decoder_output
        
        mask = src_mask.unsqueeze(1)

        attn_weights, context_vector = attention(decoder_output, memory_bank, memory_bank, mask)

        combined_input = torch.cat((decoder_output, context_vector), dim=-1)
        logits = self.hidden2output(combined_input)

        if normalize:
            logits = torch.log_softmax(logits, dim=-1)

        updated_decoder_states = (decoder_state, context_vector, prev_output)

        return logits, updated_decoder_states, attn_weights

    def evaluate_ppl(self, iterator):
        self.eval()
        total_loss = 0
        total_words = 0
        for batch in iterator:
            src = batch['src_ids']
            src_lengths = batch['src_lengths']
            tgt_in = batch['tgt_ids'][:, :-1]
            tgt_out = batch['tgt_ids'][:, 1:]
            logits = self.forward(src, src_lengths, tgt_in)
            loss = self.loss_function(logits.reshape(-1, self.V_tgt), tgt_out.reshape(-1))
            total_loss += loss.item()
            total_words += tgt_out.ne(self.padding_id_tgt).float().sum().item()
        return math.exp(total_loss / total_words)

    def train_all(self, train_iter, val_iter, epochs=10, learning_rate=0.001):
        self.train()
        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        best_validation_ppl = float('inf')
        best_model = None
        for epoch in range(epochs):
            total_words = 0
            total_loss = 0.0
            for batch in tqdm(train_iter):
                self.zero_grad()
                tgt = batch['tgt_ids']
                src = batch['src_ids']
                src_lengths = batch['src_lengths']
                tgt_in = tgt[:, :-1].contiguous()
                tgt_out = tgt[:, 1:].contiguous()
                bsz = tgt.size(0)
                logits = self.forward(src, src_lengths, tgt_in)
                loss = self.loss_function(logits.view(-1, self.V_tgt), tgt_out.view(-1))
                num_tgt_words = tgt_out.ne(self.padding_id_tgt).float().sum().item()
                total_words += num_tgt_words
                total_loss += loss.item()
                loss.div(bsz).backward()
                optim.step()

            validation_ppl = self.evaluate_ppl(val_iter)
            self.train()
            if validation_ppl < best_validation_ppl:
                best_validation_ppl = validation_ppl
                self.best_model = copy.deepcopy(self.state_dict())
            epoch_loss = total_loss / total_words
            print(f'Epoch: {epoch} Training Perplexity: {math.exp(epoch_loss):.4f} '
                  f'Validation Perplexity: {validation_ppl:.4f}')

    def predict(self, tokens, K=1, max_T=15):
        beam_searcher = BeamSearcher(self)
        
        src = torch.tensor([hf_src_tokenizer.encode(tokens)])
        src_lengths = torch.tensor([len(src[0])])

        prediction, _ = beam_searcher.beam_search(src, src_lengths, K, max_T)
        prediction = self.hf_tgt_tokenizer.decode(prediction, skip_special_tokens=True)

        return prediction
    
EPOCHS = 1 # epochs; we start with 1 epoch, but for good performance 
           # you'll want to use many more epochs, perhaps 20
LEARNING_RATE = 1e-4

# Instantiate and train classifier
model2 = AttnEncoderDecoder2(hf_src_tokenizer, hf_tgt_tokenizer,
  hidden_size    = 512,
  layers         = 3,
).to(device)

model2.train_all(train_iter, val_iter, epochs=EPOCHS, learning_rate=LEARNING_RATE)
model2.load_state_dict(model2.best_model)
print(f"Validation perplexity: {model2.evaluate_ppl(val_iter):.3f}")

query = "flights from boston to new york"



import pickle

# Assuming `model`, `hf_src_tokenizer`, and `hf_tgt_tokenizer` are defined after training
torch.save(model.state_dict(), "attn_encoder_decoder_weights.pth")
torch.save(model2.state_dict(), "attn_encoder_decoder2_weights.pth")

# Save tokenizers
with open("hf_src_tokenizer.pkl", "wb") as f_src:
    pickle.dump(hf_src_tokenizer, f_src)
with open("hf_tgt_tokenizer.pkl", "wb") as f_tgt:
    pickle.dump(hf_tgt_tokenizer, f_tgt)

model.eval()  # Set the model to evaluation mode
model2.eval()

# Use the model for prediction
def predict_flight_query1(query):
    return model.predict(query, K=1, max_T=400)

prediction1 = predict_flight_query1(query)
print("Model 1 Prediction:", prediction1)

# Use the model for prediction
def predict_flight_query2(query):
    return model2.predict(query, K=1, max_T=400)

prediction2 = predict_flight_query2(query)
print("Model 2 Prediction:", prediction2)

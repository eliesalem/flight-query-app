return_string = ""


if __name__ == "__main__":

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
    from func_timeout import func_set_timeout, FunctionTimedOut
    from torch.nn.utils.rnn import pack_padded_sequence as pack
    from torch.nn.utils.rnn import pad_packed_sequence as unpack
    from tqdm.notebook import tqdm
    from transformers import BartTokenizer, BartForConditionalGeneration

    import pickle
    from models import AttnEncoderDecoder, AttnEncoderDecoder2  # Import both model classes


    #print("hello")

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
        conn = sqlite3.connect('data/atis_sqlite.db')  # Establish the database connection
        c = conn.cursor()                              # Create a cursor
        try:
            c.execute(sql)                             # Execute the SQL query
            results = list(c.fetchall())              # Fetch all results
            return results                            # Return results
        except sqlite3.OperationalError as e:
            #print("SQL error:", e)                    # Print the error for debugging
            return -1     # Return -1 for errors
        finally:
            c.close()                                 # Always close the cursor
            conn.close()

    def execute_sql_with_timeout(sql):
        try:
            return execute_sql(sql)
        except FunctionTimedOut:
            return -1


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
            print("DB Result:\n\n", predicted_result[:10], "\n")

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

    query_tree = parse_tree(query)
    #query_tree.pretty_print()
    query_sql = interpret(query_tree, atis_augmentations)
    #return_string += "Correct query SQL: " + query_sql
    print("<b>Correct query SQL: </b>", query_sql)
    query_result = execute_sql_with_timeout(query_sql)
    #return_string += "<br>Correct DB Result: " + str(query_result[:10])
    print("<br><b>The IDs of the requested flights are: </b>", query_result[:10])



    # Load tokenizers
    with open("hf_src_tokenizer.pkl", "rb") as f_src:
        hf_src_tokenizer = pickle.load(f_src)
    with open("hf_tgt_tokenizer.pkl", "rb") as f_tgt:
        hf_tgt_tokenizer = pickle.load(f_tgt)

    # Re-instantiate model 1
    model = AttnEncoderDecoder(
        hf_src_tokenizer=hf_src_tokenizer,
        hf_tgt_tokenizer=hf_tgt_tokenizer,
        hidden_size=2,  # Match training parameters
        layers=1
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Load weights for model 1
    model.load_state_dict(torch.load("attn_encoder_decoder_weights.pth"))
    model.eval()

    # Re-instantiate model 2
    model2 = AttnEncoderDecoder2(
        hf_src_tokenizer=hf_src_tokenizer,
        hf_tgt_tokenizer=hf_tgt_tokenizer,
        hidden_size=512,  # Match training parameters
        layers=3
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Load weights for model 2
    model2.load_state_dict(torch.load("attn_encoder_decoder2_weights.pth"))
    model2.eval()

    # Prediction functions
    def predict_flight_query(query):
        return model.predict(query)

    def predict_flight_query_with_model2(query):
        return model2.predict(query)

    model1_query = predict_flight_query(query)
    #return_string += "<br><br>Model 1 prediction: " + model1_query
    print("<br><br><b>Model 1 prediction: </b>", model1_query)

    model1_db_result = execute_sql_with_timeout(model1_query)

    if model1_db_result != -1:
        #return_string += "<br>Model 1 DB Result: " + str(model1_db_result[:10])
        print("<br><b>Model 1 DB Result: </b>", model1_db_result[:10])
    else:
        #return_string += "<br>Error: Model 1 query is not valid sql."
        print("<br><b>Model 1 DB Result: </b> Error: Model 1 query is not valid sql.")

    model2_query = predict_flight_query_with_model2(query)
    #return_string += "<br><br>Model 2 Prediction: " + model2_query
    print("<br><br><b>Model 2 Prediction: </b>", model2_query)

    model2_db_result = execute_sql_with_timeout(model2_query)

    if model2_db_result != -1:
        #return_string += "<br>Model 2 DB Result: " + str(model2_db_result[:10])
        print("<br><b>Model 2 DB Result: </b>", model2_db_result[:10])
    else:
        #return_string += "<br>Error: Model 2 query is not valid sql."
        print("<br><b>Model 2 DB Result: </b>Error: Model 2 query is not valid sql.")

    #print("python is running and return string is updating", return_string)

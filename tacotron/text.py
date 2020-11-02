""" adapted from https://github.com/keithito/tacotron """

import re
from itertools import islice
import importlib_resources

SYMBOLS = [
    "_",  # padding
    "~",  # end of sentence
    " ",
    "!",
    ",",
    ".",
    "?",
    "AA",
    "AA0",
    "AA1",
    "AA2",
    "AE",
    "AE0",
    "AE1",
    "AE2",
    "AH",
    "AH0",
    "AH1",
    "AH2",
    "AO",
    "AO0",
    "AO1",
    "AO2",
    "AW",
    "AW0",
    "AW1",
    "AW2",
    "AY",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "EH0",
    "EH1",
    "EH2",
    "ER",
    "ER0",
    "ER1",
    "ER2",
    "EY",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH",
    "IH0",
    "IH1",
    "IH2",
    "IY",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OW0",
    "OW1",
    "OW2",
    "OY",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]

symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}
id_to_symbol = {i: s for i, s in enumerate(SYMBOLS)}

abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1].upper())
    for x in [
        ("mrs", "missis"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
        ("etc", "etcetera"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in abbreviations:
        text = re.sub(regex, replacement, text)
    return text


alternate_pronunciation_regex = re.compile(r"(?<=\w)\((\d)\)")
parentheses_regex = re.compile(r"(?<=[.,!?] )[\(\[]|[\)\]](?=[.,!?])|^[\(\[]|[\)\]]$")
dash_regex = re.compile(r"(?<=[.,!?] )-- ")


def replace_symbols(text, lang="en"):
    # replace semi-colons and colons with commas
    text = text.replace(";", ",")
    text = text.replace(":", ",")

    # replace dashes with commas
    text = re.sub(dash_regex, "", text)
    text = text.replace(" --", ",")
    text = text.replace(" - ", ", ")

    # split hyphenated words
    text = text.replace("-", " ")

    # use {#} to indicate alternate pronunciations
    text = re.sub(alternate_pronunciation_regex, r"{\1}", text)

    # replace parentheses with commas
    text = re.sub(parentheses_regex, "", text)
    text = text.replace(")", ",")
    text = text.replace(" (", ", ")
    text = text.replace("]", ",")
    text = text.replace(" [", ", ")
    return text


def clean(text):
    text = text.upper()
    text = expand_abbreviations(text)
    text = replace_symbols(text)
    return text


tokenizer_regex = re.compile(r"[\w\{\}']+|[.,!?]")


def tokenize(text):
    return re.findall(tokenizer_regex, text)


def load_cmudict():
    cmudict = dict()
    dict_ref = importlib_resources.files("tacotron.dictionary").joinpath(
        "cmudict-0.7b.txt"
    )
    with open(dict_ref, encoding="ISO-8859-1") as file:
        for line in islice(file, 126, 133905):
            word, pronunciation = line.strip().split("  ")
            word = re.sub(alternate_pronunciation_regex, r"{\1}", word)
            cmudict[word] = pronunciation
    return cmudict


def parse_text(text, cmudict):
    text = clean(text)
    symbols, out_of_vocabulary = list(), set()
    for word in tokenize(text):
        word_symbols = cmudict.get(word, word if word in ".,!?" else None)
        if word_symbols is not None:
            symbols.append(word_symbols.split(" "))
        else:
            out_of_vocabulary.add(word)

    if out_of_vocabulary:
        raise KeyError(
            f"Please add the following words to the pronunciation dictionary: {', '.join(out_of_vocabulary)}"
        )

    symbols = [symbol for word in symbols for symbol in (word, [" "])]
    symbols.append(["~"])
    return symbols


def symbols_to_id(symbols):
    return [symbol_to_id[symbol] for word in symbols for symbol in word]


def text_to_id(text, cmudict):
    symbols = parse_text(text, cmudict)
    return symbols_to_id(symbols)
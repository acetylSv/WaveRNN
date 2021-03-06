from utils.files import get_files
from pathlib import Path
from typing import Union


def ljspeech(path: Union[str, Path]):
    #csv_file = get_files(path, extension='.csv')
    
    #assert len(csv_file) == 1
    assert path is not None

    text_dict = {}

    with open(path, encoding='utf-8') as f :
        for line in f :
            split = line.split('|')
            text_dict[split[0]] = split[-1]

    return text_dict

def DaAi(path: Union[str, Path]):
    txt_files = get_files(path, extension='.txt')

    assert len(txt_files) != 0

    text_dict = {}

    for txt_file in txt_files:
        with open(txt_file, encoding='utf-8') as f :
            for line in f :
                # some preprocessing here?
                line = line.strip().replace(' ', '')
                text_dict[txt_file.name.replace('.txt', '')] = line

    return text_dict

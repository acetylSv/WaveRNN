""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
#from utils.text import cmudict

#_pad = '_'
#_punctuation = '!\'(),.:;? '
#_special = '-'
#_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

## Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ['@' + s for s in cmudict.valid_symbols]

## Export all symbols:
#symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet


# For DaAi
#from utils import hparams as hp
#from utils.paths import Paths
#paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

import pickle

_pad = '_'
_unk = '?'
zh_TW_char_dict = {}
zh_TW_char = [_pad] + [_unk]

path = '/storage/ranmei/DaAi/data/text_dict.pkl'
with open(path, 'rb') as f:
    all_txt = pickle.load(f)
    for txt_name in all_txt:
        line = all_txt[txt_name].strip()
        for char in line:
            if char not in zh_TW_char_dict:
                zh_TW_char_dict[char] = 1
            else:
                zh_TW_char_dict[char] += 1

filter_out = ['〈', '〉','《', '》', '「', '」', '(', ')', ':','（', '）', '.', '、', 'ㄧ', '‧']
from operator import itemgetter
for k, v in sorted(zh_TW_char_dict.items(), key=itemgetter(1)):
    if v >= 5 and k not in filter_out:
        zh_TW_char += k

symbols = zh_TW_char

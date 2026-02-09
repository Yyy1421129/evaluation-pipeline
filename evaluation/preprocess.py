import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from normalization.asr.asr_simple_tn import asr_num2words
class Preprocessor:
    def __init__(self, lang="en", map_dir=None, debug=False):
        self.lang = lang
        self.map_dir = map_dir or "normalization/asr/asr_simple_tn_rules"
        self.debug = debug

    def normalize(self, text):
        return asr_num2words(text, self.lang, self.map_dir, self.debug)
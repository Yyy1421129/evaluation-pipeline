import unicodedata
import re

ALL_PUNCTS = set([
    '!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/',
    ':', ';', '=', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '}', '~',
    '、', '。', '！', '，', '；', '？', '：', '「', '」', '︰', '『', '』', '《', '》'
])

SPACELIST = [' ', '\t', '\r', '\n']


def stripoff_tags(text):
    if not text:
        return ''
    chars = []
    i = 0
    T = len(text)
    while i < T:
        if text[i] == '<':
            while i < T and text[i] != '>':
                i += 1
            i += 1
        else:
            chars.append(text[i])
            i += 1
    return ''.join(chars)


def remove_all_puncts(text):
    if not text:
        return ""
    return ''.join([c for c in text if c not in ALL_PUNCTS or c in SPACELIST])


def normalize_text(text, case_sensitive=False, remove_tag=True):
    if not text:
        return ""
    
    if remove_tag:
        text = stripoff_tags(text)
    
    text = remove_all_puncts(text)
    
    if not case_sensitive:
        text = text.upper()
    
    text = ' '.join(text.split())
    
    return text.strip()

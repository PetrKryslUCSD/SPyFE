# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 07:26:37 2017

@author: PetrKrysl
"""
import io
from tkinter import Tk
r = Tk()
r.withdraw()
clipping = r.selection_get(selection = "CLIPBOARD")

import tokenize

def listtok(s):
    """Tweak Matlab  expressions.
    """
    result = []
    g = tokenize.generate_tokens(io.StringIO(s).readline)   # tokenize the string
    in_deref=False
    in_comment = False
    last_was_name = False
    for tup in g:
        if tup.string == '%':
            in_comment = True
            result.extend([
                (tokenize.COMMENT, "# ")
            ])
        else:
            if not in_comment:
                if tup.exact_type == tokenize.LPAR and last_was_name:
                    in_deref=True
                    result.extend([
                        (tokenize.OP, '[')
                    ])
                elif  in_deref:
                    if tup.exact_type == tokenize.RPAR:
                        in_deref=False
                        result.extend([
                            (tokenize.OP, ']')
                        ])
                    elif tup.type == tokenize.NUMBER:
                        num=int(tup.string)-1
                        result.extend([
                            (tokenize.NUMBER, str(num))
                        ])
                    elif tup.type == tokenize.NAME:
                        result.extend([
                            (tokenize.NUMBER, tup.string+"-1")
                        ])
                    else:
                        result.append((tup.type, tup.string))
                else:
                    if tup.exact_type == tokenize.SEMI:
                        pass
                    else:
                        if tup.exact_type == tokenize.NAME:
                            last_was_name = True
                        else:
                            last_was_name = False
                        result.append((tup.type, tup.string))
            else:
                result.append((tup.type, tup.string))
        if tup.exact_type == tokenize.NEWLINE:
            in_comment = False
    return tokenize.untokenize(result)

print(listtok(clipping))


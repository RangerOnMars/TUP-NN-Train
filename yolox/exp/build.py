#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import yolox
import importlib
import os
import sys


def get_exp_by_file(exp_file):
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        print(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()
        #print(exp)
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


def get_exp_by_name(exp_name):

    yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
    filedict = {
        "nano-poly": "nano-poly.py",
        "nano-polyv2": "nano-polyv2.py",
        "shufflev2-nano": "shufflev2-nano.py",
        "tiny-poly": "tiny-poly.py",
        "xs-poly": "xs-poly.py",
        "xs-poly-DW": "xs-poly-DW.py",
    }
    filename = filedict[exp_name]
    exp_path = os.path.join(yolox_path, "exps", "default", filename)
    return get_exp_by_file(exp_path)


def get_exp(exp_file, exp_name):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file)
    else:
        return get_exp_by_name(exp_name)

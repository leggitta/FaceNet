#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 20:23:13 2020

@author: blackhawk
"""

def ChangeName(f):
    return "{"+"{{url_for('static', filename = '{}')}}".format(f)+"}"



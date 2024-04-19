"""
  Copyright 2020 ETH Zurich, Secure, Reliable, and Intelligent Systems Lab

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import re
from constraints import *

def clean_string(string):
    return string.replace('\n', '')

def label_index(label):
    return int(clean_string(label)[1:])

def get_constraints_for_dominant_label(label, failed_labels):
    c = Constraints()
    c.add(Constraint(label, "max"))
    return c

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def get_constraints_from_file(file):
    constraints = Constraints()
    lines = open(file, 'r').readlines()  # AND

    num_labels = int(lines[0])
    for index in range(1, len(lines)):
        elements = re.split(' +', lines[index])
        i = 0
        labels = []  # OR
        while elements[i].startswith('y'):
            labels.append(label_index(elements[i]))
            i += 1

        constraint = clean_string(elements[i])
        i += 1

        if constraint in ["min", "max", "notmin", "notmax"]:
            constraints.add(Constraint(label, constraint))
        elif constraint in [">", "<", ">=", "<="]:
            if isfloat(elements[i]):
                constraints.add(Constraint(label, constraint, float(elements[i])))
            else:
                constraints.add(Constraint(label, constraint, label_index(elements[i])))

    return constraints


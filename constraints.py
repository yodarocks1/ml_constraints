import numpy as np

class Constraint:
    """
       ------------------------------------------------------------------------
    Class Attributes:
      INVERT_MAP  Running INVERT_MAP[constraint] inverts the constraint
      F           Calling F[constraint](a, b) runs the constraint as a function
                        (Only valid for Comparison Constraints)
        
    Object Attribute Types:
      constraint  *ComparisonConstraint*  or  *MaxConstraint*
      labels       Index                  or   list<Index> (length >= 1)
      other        Index or Constant      or   `None`
      other_const  boolean                or   `False`
      repr         string
      vnnlib       boolean

    Types used:
      Index: int
      Constant: string
        format: type<float> + "f"
      ComparisonContraint: string
        one of: '>', '>=', '<', '<='
      MaxConstraint: string
        one of: 'max', 'min', 'notmax', 'notmin'
    
    Valid constraints:
      >        implies   LABEL_1 >  OTHER
      >=       implies   LABEL_1 >= OTHER
      <        implies   LABEL_1 <  OTHER
      <=       implies   LABEL_1 <= OTHER
      max      implies   max(LABELS) in [LABEL_1, LABEL_2, LABEL_3]
      min      implies   min(LABELS) in [LABEL_1, LABEL_2, LABEL_3]
      notmax   implies   max(LABELS) not in [LABEL_1, LABEL_2, LABEL_3]
      notmin   implies   min(LABELS) not in [LABEL_1, LABEL_2, LABEL_3]
    """
    
    INVERT_MAP = {
        "min": "notmin",
        "max": "notmax",
        "notmin": "min",
        "notmax": "max",
        ">": "<=",
        "<": ">=",
        ">=": "<",
        "<=": ">"
    }
    """Map of constraint values to their inverted form."""
    
    F = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
    }
    """Returns the constraint value as an anonymous function.
        (Only valid for comparison functions: >, <, >=, <=)
    """

    def __invert__(self) -> Constraint:
        raise NotImplementedError(f"Constraint type {type(self)} did not implement __invert__(self) -> Constraint")
    
    def __repr__(self) -> str:
        raise NotImplementedError(f"Constraint type {type(self)} did not implement __repr__(self) -> str")

    def percent_true(self, lower_bound, upper_bound) -> float:
        raise NotImplementedError(f"Constraint type {type(self)} did not implement percent_true(self, lower_bound, upper_bound) -> float")
        
    def exact_true(self, values) -> bool:
        raise NotImplementedError(f"Constraint type {type(self)} did not implement exact_true(self, values) -> bool")
        
    def is_vnnlib(self, coerce=False) -> bool:
        raise NotImplementedError(f"Constraint type {type(self)} did not implement is_vnnlib(self, coerce=False) -> bool")

    def force_vnnlib(self) -> Constraint:
        raise NotImplementedError(f"Constraint type {type(self)} did not implement force_vnnlib(self) -> Constraint")
    
    def __new__(self, labels, constraint, other=None):
        pass
    
    def is_vnnlib(self, force=False):
        """Returns whether the constraint is immediately representable in VNNLIB format.
        
        Keyword arguments:
        force   Allow coercion. (default false)
                    TODO: define 'coercion'
        """
        if self.constraint == ">=" and not force:
            return False
        if self.other is not None:
            if self.constraint == "<=" and type(self.other) is not float:
                return False
            if self.constraint != "<=" and type(self.other) is float:
                return False
        return True

    def force_vnnlib(self):
        """Return this constraint in VNNLIB format
        Errors:
        ValueError  Constraint could not be coerced into VNNLIB format.
                        See is_vnnlib() with force=true for further information
        """
        if self.is_vnnlib():
            return self
        elif not self.is_vnnlib(force=True):
            raise ValueError(f"Cannot convert constraint `{self.repr}` to vnnlib format")
        return Constraint(self.labels, self.constraint.replace("=", ""), other=self.other)

    def __call__(self, values_or_lower_bound, upper_bound=None, min_percent=None):
        if upper_bound is None:
            if min_percent is not None:
                raise TypeError("TypeError: __call__() got an unexpected keyword argument 'min_percent'")
            values = values_or_lower_bound
            # __call__(self, values)
            return self.exact_true(values)
        else:
            lower_bound = values_or_lower_bound
            # __call__(self, lower_bound, upper_bound, min_percent=None)
            percent = self.percent_true(lower_bound, upper_bound)
            if min_percent is None:
                return percent
            return percent >= min_percent

class ComparisonConstraint(Constraint):
    VALID_CONSTRAINTS = ('>', '>=', '<', '<=')
    INVERT_MAP = {
        '>': '<=',
        '>=': '<',
        '<': '>=',
        '<=': '>',
    }
    F = {
        '>': lambda a, b: a > b,
        '<': lambda a, b: a < b,
        '>=': lambda a, b: a >= b,
        '<=': lambda a, b: a <= b,
    }
    def __init__(self, label, constraint, other):
        if constraint not in ComparisonConstraint.VALID_CONSTRAINTS:
            raise ValueError(f"Bad constraint `{constraint}`")
        elif other is None:
            raise ValueError(f"Comparison constraints (like `{constraint}`) require a label or value on the right")

        try:
            self.label = int(label)
        except ValueError:
            raise ValueError(f"Invalid label index {label}")
            
        if other.endswith("f"):
            try:
                self.other = float(other[:-1])
                self.other_const = True
            except ValueError:
                raise ValueError(f"Invalid constant value {other[:-1]}; constant values are indicated by terminating with 'f' (e.g. 4f)")
        elif not other.isdigit():
            raise ValueError(f"Invalid label index {other}; constant values are indicated by terminating with 'f' (e.g. y2 > 4f)")
        else
            self.other = int(other)
            self.other_const = False

        self.vnnlib = (self.other_const and constraint == ">=") or (not self.other_const and constraint in [">", "<"])

    def __invert__(self) -> Constraint:
        inverted_constraint = ComparisonConstraint.INVERT_MAP[self.constraint]
        if self.other_const:
            return ComparisonConstraint(self.label, inverted_constraint, self.other + "f")
        else:
            return ComparisonConstraint(self.label, inverted_constraint, self.other)
    
    def __repr__(self) -> str:
        if self.other_const:
            return f"y{self.label} {constraint} {self.other}f"
        else:
            return f"y{self.label} {constraint} y{self.other}"

    def percent_true(self, lower_bound, upper_bound) -> float:
        v1_low = lower_bound[self.labels[0]]
        v1_high = upper_bound[self.labels[0]]
        if not self.other_const:
            v2_low = lower_bound[self.other]
            v2_high = upper_bound[self.other]
        else:
            v2_low = self.other
            v2_high = self.other

        # Invert less-than so we only have to check once (keep `=`, if present)
        if '<' in self.constraint[0]: # '<' or '<='
            v1_low, v1_high, v2_low, v2_high = v2_low, v2_high, v1_low, v1_high # swap v1 and v2
            f = Constraint.F[self.constraint.replace('<', '>')] # '<' -> '>', '<=' -> '>='

        # ALL greater: low >(=) high
        if f(v1_low, v2_high):
            return 1.0
        # SOME greater: high >(=) low
        elif f(v1_high, v2_low):
            v1_range = v1_high - v1_low
            v2_range = v2_high - v2_low
            overlap = v1_high - v2_low
            return (overlap / v1_range) * (overlap / v2_range)
        # NONE greater
        else:
            return 0.0
            
    def exact_true(self, values) -> bool:
        v1 = values[self.labels[0]]
        v2 = self.other if self.other_const else values[self.other]
        return ComparisonConstraint.F[self.constraint](v1, v2)

    def is_vnnlib(self, coerce=False) -> bool:
        """Returns whether the constraint is immediately representable in VNNLIB format.
        Allowed in VNNLIB, by type of Other:
           ____|_Const_|_Index_|
          | >  |  No   |  Yes  |
          | >= |  No   |  No   |
          | <  |  No   |  Yes  |
          | <= |  Yes  |  No   |
        Allowed in VNNLIB, by type of Other, with coercion:
           ____|_Const_|_Index_|
          | >  |  No   |  Yes  |
          | >= |  No   |  Yes  |
          | <  |  Yes  |  Yes  |
          | <= |  Yes  |  Yes  |
        
        Keyword arguments:
          coerce   Allow >= and <= to be coerced to > and < (respectively), or vice versa. (default False)
        """
        if '=' in self.constraint and not self.other_const:  # Index; >=, <=
            return coerce # VNNLIB does not allow >= or <= with indices
                          #    These can be coerced to > and <, respectively
        elif '>' in self.constraint and self.other_const:    # Const; >=, >
            return False  # VNNLIB does not allow >= or > with constants
                          #    Coercion does nothing
        elif self.constraint == '<' and self.other_const:    # Const; <
            return coerce # VNNLIB does not allow < with constants
                          #    This can be coerced to >=
        else:                                                # Index; >, <
            return True                                      # Const; <=

    def force_vnnlib(self) -> Constraint:
        """Return this constraint in VNNLIB format
        *WARNING - This may change equality inclusion (e.g. >= to >, or < to <=)!
        
        Errors:
          ValueError  Constraint could not be coerced into VNNLIB format.
                        See is_vnnlib() with coerce=true for further information
        """
        if self.is_vnnlib():
            return self
        elif not self.is_vnnlib(coerce=True):
            # > and >= with constants cannot be coerced into VNNLIB format
            raise ValueError(f"Cannot convert constraint `{self.repr}` to vnnlib format")
        if self.other_const:
            # VNNLIB does not allow < comparisons with constants
            return ComparisonConstraint(self.label, self.constraint.replace('<', '<='), self.other + "f")
        else:
            # VNNLIB does not allow <= or >= comparisons with indices
            return ComparisonConstraint(self.label, self.constraint.replace("=", ""), self.other)

class MaxConstraint(Constraint):
    VALID_CONSTRAINTS = ('max', 'min', 'notmax', 'notmin')
    INVERT_MAP = {
        'max': 'notmax',
        'min': 'notmin',
        'notmax': 'max',
        'notmin': 'min',
    }
    F = {
        'max': _F(np.argmax),
        'min': _F(np.argmin),
        'notmax': _F(np.argmax, True),
        'notmin': _F(np.argmin, True),
    }
    @staticmethod
    def _F(func, invert=False):
        if invert:
            return lambda labels, values: func(values) not in labels
        else:
            return lambda labels, values: func(values) in labels
    
    def __init__(self, labels, constraint):
        if constraint not in MaxConstraint.VALID_CONSTRAINTS:
            raise ValueError(f"Bad constraint `{constraint}`")
        elif len(labels) == 0:
            raise ValueError(f"Max Constraints (like `{constraint}`) require at least one label")

        self.labels = []
        for label in labels:
            try:
                self.labels.append(int(label))
            except ValueError:
                raise ValueError(f"Invalid label index {label}")

    def __invert__(self) -> Constraint:
        return MaxConstraint(self.labels, MaxConstraint.INVERT_MAP[self.constraint])
    
    def __repr__(self) -> str:
        return " ".join(map(lambda label: f"y {label}", self.labels)) + " " + self.constraint

    def percent_true(self, lower_bound, upper_bound) -> float:
        # TODO: Check against what the internet says @ https://stackoverflow.com/questions/78332169
        chosen = self.labels
        unchosen = list(filter(lambda x: x not in chosen, range(len(lower_bound))))

        f = max if "max" in self.constraint else min

        chosen_low = f(map(lambda x: lower_bound[x], chosen))
        chosen_high = f(map(lambda x: upper_bound[x], chosen))
        unchosen_low = f(map(lambda x: lower_bound[x], unchosen))
        unchosen_high = f(map(lambda x: upper_bound[x], unchosen))

        # SIMPLIFIES TO:

        ########################             ########################
        #           ╱C=U       #             #           ╱C=U       #
        # C        ╱           #             # C        ╱           #
        #    ░░░░░▟█████       #  █: MIN     #    █████▛░░░░░       #  █: MAX
        #    ░░░░▟██████ Area  #  ░: NOTMIN  #    ████▛░░░░░░ Area  #  ░: NOTMAX
        #    ░░░▟███████  is   #             #    ███▛░░░░░░░  is   #
        #    ░░▟████████ Prob- #             #    ██▛░░░░░░░░ Prob- #
        #    ░▟█████████ abil- #             #    █▛░░░░░░░░░ abil- #
        # c  ▟██████████ ity   #             # c  ▛░░░░░░░░░░ ity   #
        #   ╱                  #             #   ╱                  #
        #  ╱                   #             #  ╱                   #
        # ╱  u         U       #             # ╱  u         U       #
        ########################             ########################
        
        c, C = chosen_low, chosen_high
        u, U = unchosen_low, unchosen_high
        total_area = (C-c)*(U-u)

        if c >= U:
            max_result = 1
        elif u >= C:
            max_result = 0
        elif U >= C: # C > u
            max_result = 0.5 * (C-u) * (C-c) # Triangle
            max_result /= total_area # Normalize to percentage
        else: # C >= U, U >= c
            max_result = 0.5 * (U-c) * (U-u) # Triangle
            max_result += (C-U) * (U-u) # Rectangle
            max_result /= total_area # Normalize to percentage

        if self.constraint == "min" or self.constraint == "notmax":
            return 1 - max_result
        else:
            return max_result
        
    def exact_true(self, values) -> bool:
        return MaxConstraint.F[self.constraint](self.labels, values)
        
    def is_vnnlib(self, coerce=False) -> bool:
        pass

    def force_vnnlib(self) -> Constraint:
        pass

class Constraints:
    @classmethod
    def from_text(cls, text):
        c = cls()
        if type(text) is str:
            text = text.split("\n")
        for line in text:
            parts = line.split(" ")
            labels = []
            i = 0
            while parts[i] not in Constraint.INVERT_MAP:
                labels.append(parts[i])
                i += 1
            if len(parts) == i + 1: # constraint is last element
                c.add(Constraint(labels, parts[i]))
            else:
                c.add(Constraint(labels, parts[i], parts[i+1]))
        return c
    @classmethod
    def from_constraint_file(cls, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        return cls.from_text(lines)
    @classmethod
    def from_label(cls, label):
        c = cls()
        c.add(Constraint((label, ), "max"))
        return c


    def __init__(self):
        self.constraints = []

    def add(self, *constraints):
        self.constraints.extend(constraints)

    def is_vnnlib(self, force=False):
        for constraint in self.constraints:
            if not constraint.is_vnnlib(force=force):
                return False
        return True

    def force_vnnlib(self):
        c = Constraints()
        for constraint in self.constraints:
            c.add(constraint.force_vnnlib())
        return c

    def __invert__(self):
        c = Constraints()
        for constraint in self.constraints:
            c.add(~constraint)
        return c

    def __repr__(self):
        return '\n'.join(map(repr, self.constraints))

    def __call__(self, values, upper_bound=None, min_percent=None):
        if upper_bound is None:
            for constraint in self.constraints:
                if not constraint(values):
                    return False
            return True
        elif len(self.constraints) == 0:
            return 1
        elif min_percent is not None:
            for constraint in self.constraints:
                if not constraint(values, upper_bound, min_percent=min_percent):
                    return False
            return True
        else:
            total = 0
            for constraint in self.constraints:
                total += constraint.percent_true(values, upper_bound)
            else:
                return total / len(self.constraints)


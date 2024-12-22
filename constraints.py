from typing import Tuple
import numpy as np

class Label(int):
    def __new__(cls, v):
        if type(v) is str:
            if not v.startswith("y"):
                raise ValueError(f"Invalid label index {v}; label indices must start with 'y' (e.g. y4)")
            try:
                return super(Label, cls).__new__(cls, v[1:])
            except ValueError:
                raise ValueError(f"Invalid label index {v}")
        elif issubclass(type(v), int):
            return super(Label, cls).__new__(cls, v)
        else:
            raise TypeError(f"Invalid label index {v} of type {type(v).__name__}; expecting int or str")
class Constant(float):
    def __new__(cls, v):
        if type(v) is str:
            if not v.endswith("f"):
                raise ValueError(f"Invalid constant {v}; constants must end in 'f' (e.g. 8f or -2.24e8f)")
            try:
                return super(Constant, cls).__new__(cls, v[:-1])
            except ValueError:
                raise ValueError(f"Invalid constant {v}")
        elif issubclass(type(v), float):
            return super(Constant, cls).__new__(cls, v)
        else:
            raise TypeError(f"Invalid constant {v} of type {type(v).__name__}; expecting float or str")
def LabelOrConstant(v):
    t = type(v)
    if issubclass(t, int):
        return Label(v)
    elif issubclass(t, float):
        return Constant(v)
    elif t is str:
        try:
            return Label(v)
        except ValueError:
            pass
        try:
            return Constant(v)
        except ValueError:
            pass
        raise ValueError(f"Invalid constant or label index {v}; constants must end with 'f' (e.g. 8f or -2.24e8f); label indices must start with 'y' (e.g. y4)")
    else:
        raise TypeError(f"Invalid constant or label index {v} of type {type(v).__name__}; expecting int, float, or str")

class Constraint:
    VALID_CONSTRAINTS = () # Initialized later
    
    def __new__(cls, labels, constraint, *args, **kwargs):
        """Creation of a Constraint will be delegated to a subclass"""
        if cls is Constraint:
            if constraint in ComparisonConstraint.VALID_CONSTRAINTS:
                return super(Constraint, cls).__new__(ComparisonConstraint)
            elif constraint in MaxConstraint.VALID_CONSTRAINTS:
                return super(Constraint, cls).__new__(MaxConstraint)
            else:
                raise ValueError(f"Bad constraint `{constraint}`")
        else:
            return super(Constraint, cls).__new__(cls)

    def __invert__(self) -> 'Constraint':
        """Inverts the constraint.
        Calls to `inverted.percent_true(...)` will always return `1 - self.percent_true(...)`
        Calls to `inverted.exact_true(...)` will always return `not self.exact_true(...)`
        """
        raise NotImplementedError(f"Constraint type {type(self)} did not implement __invert__(self) -> Constraint")
    
    def __repr__(self) -> str:
        """Returns the constraint in proper constraints-file format.
        All labels are represented by yN, where N is their index.
        All constants are represented by Nf, where N is any float value.
        """
        raise NotImplementedError(f"Constraint type {type(self)} did not implement __repr__(self) -> str")

    def percent_true(self, lower_bound, upper_bound) -> float:
        """Returns the percentage of values between lower_bound and upper_bound that follow the constraint."""
        raise NotImplementedError(f"Constraint type {type(self)} did not implement percent_true(self, lower_bound, upper_bound) -> float")
        
    def exact_true(self, values) -> bool:
        """Returns whether the values given follow the constraint."""
        raise NotImplementedError(f"Constraint type {type(self)} did not implement exact_true(self, values) -> bool")
        
    def is_vnnlib(self, coerce=False) -> bool:
        """Returns whether the constraint is immediately representable in VNNLIB format."""
        raise NotImplementedError(f"Constraint type {type(self)} did not implement is_vnnlib(self, coerce=False) -> bool")

    def force_vnnlib(self) -> 'Constraint':
        """Return this constraint in VNNLIB format.
        *WARNING - This may change equality inclusion (e.g. >= to >, or < to <=)!

        Exceptions:
          ValueError  Constraint could not be coerced into VNNLIB format.
                        See is_vnnlib() with coerce=true for further information
        """
        raise NotImplementedError(f"Constraint type {type(self)} did not implement force_vnnlib(self) -> Constraint")

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
        elif other is None or type(other) is not str:
            raise ValueError(f"Comparison constraints (like `{constraint}`) require a label or value on the right")
        elif type(label) is not str:
            if len(label) != 1:
                raise TypeError(f"Expected an iterable of length 1, got {type(label).__name__} of length {len(label)}")
            else:
                label = label[0]
                if type(label) is not str:
                    raise TypeError(f"iterable[0]: Expected a str, got {type(label).__name__}")

        self.label = Label(label)
        self.constraint = constraint
        self.other = LabelOrConstant(other)

        self.other_const = type(other) is Constant

        self.vnnlib = (self.other_const and constraint == ">=") or (not self.other_const and constraint in [">", "<"])

    def __invert__(self) -> Constraint:
        """Inverts the constraint.
        Calls to `inverted.percent_true(...)` will always return `1 - self.percent_true(...)`
        Calls to `inverted.exact_true(...)` will always return `not self.exact_true(...)`

        This is done by returning a ComparisonConstraint that is the proper inverse of this one.
          E.g. `x > y` => `x <= y`
        """
        inverted_constraint = ComparisonConstraint.INVERT_MAP[self.constraint]
        if self.other_const:
            return ComparisonConstraint(self.label, inverted_constraint, self.other + "f")
        else:
            return ComparisonConstraint(self.label, inverted_constraint, self.other)
    
    def __repr__(self) -> str:
        """Returns the constraint in proper constraints-file format.
        All labels are represented by yN, where N is their index.
        All constants are represented by Nf, where N is any float value.
        """
        if self.other_const:
            return f"y{self.label} {constraint} {self.other}f"
        else:
            return f"y{self.label} {constraint} y{self.other}"

    def percent_true(self, lower_bound, upper_bound) -> float:
        """Returns the percentage of values between lower_bound and upper_bound that follow the constraint."""
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
        """Returns whether the values given follow the constraint."""
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
        """Return this constraint in VNNLIB format.
        *WARNING - This may change equality inclusion (e.g. >= to >, or < to <=)!
        
        Exceptions:
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
    # Not @staticmethod because we don't need to access this outside of the static context
    def _F(func, invert=False):
        if invert:
            return lambda labels, values: func(values) not in labels
        else:
            return lambda labels, values: func(values) in labels
    F = {
        'max': _F(np.argmax),
        'min': _F(np.argmin),
        'notmax': _F(np.argmax, True),
        'notmin': _F(np.argmin, True),
    }
    
    def __init__(self, labels, constraint):
        if constraint not in MaxConstraint.VALID_CONSTRAINTS:
            raise ValueError(f"Bad constraint `{constraint}`")
        elif len(labels) == 0:
            raise ValueError(f"Max Constraints (like `{constraint}`) require at least one label")

        self.labels = []
        for label in labels:
            self.labels.append(Label(label))
        self.constraint = constraint

    def __invert__(self) -> Constraint:
        """Inverts the constraint.
        Calls to `inverted.percent_true(...)` will always return `1 - self.percent_true(...)`
        Calls to `inverted.exact_true(...)` will always return `not self.exact_true(...)`

        This is done by returning a ComparisonConstraint that is the proper inverse of this one.
          E.g. `y2 y4 min` => `y2 y4 notmin`
        """
        return MaxConstraint(self.labels, MaxConstraint.INVERT_MAP[self.constraint])
    
    def __repr__(self) -> str:
        """Returns the constraint in proper constraints-file format.
        All labels are represented by yN, where N is their index.
        All constants are represented by Nf, where N is any float value.
        """
        return " ".join(map(lambda label: f"y {label}", self.labels)) + " " + self.constraint

    def percent_true(self, lower_bound, upper_bound) -> float:
        """Returns the percentage of values between lower_bound and upper_bound that follow the constraint.
        c = chosen low   | C = chosen high
        u = unchosen low | U = unchosen high
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
        """
        # TODO: Check against what the internet says @ https://stackoverflow.com/questions/78332169
        chosen = self.labels
        unchosen = list(filter(lambda x: x not in chosen, range(len(lower_bound))))

        f = max if "max" in self.constraint else min

        chosen_low = f(map(lambda x: lower_bound[x], chosen))
        chosen_high = f(map(lambda x: upper_bound[x], chosen))
        unchosen_low = f(map(lambda x: lower_bound[x], unchosen))
        unchosen_high = f(map(lambda x: upper_bound[x], unchosen))

        # See docstring:
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
        """Returns whether the values given follow the constraint."""
        return MaxConstraint.F[self.constraint](self.labels, values)
        
    def is_vnnlib(self, coerce=False) -> bool:
        """Returns whether the constraint is immediately representable in VNNLIB format.
        All of max, min, notmax, and notmin are allowed in VNNLIB. Therefore,
            MaxConstraint.is_vnnlib() will always return True.
        """
        return True

    def force_vnnlib(self) -> Constraint:
        """Return this constraint in VNNLIB format.
        All of max, min, notmax, and notmin are allowed in VNNLIB. Therefore,
            MaxConstraint.force_vnnlib() will never change the constraint.
        """
        return self
Constraint.VALID_CONSTRAINTS = (*ComparisonConstraint.VALID_CONSTRAINTS, *MaxConstraint.VALID_CONSTRAINTS)

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
            while parts[i] not in Constraint.VALID_CONSTRAINTS:
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

    def percent_true(self, lower_bound, upper_bound):
        if len(self.constraints) == 0:
            return 1
        result = 1
        for constraint in self.constraints:
            # Rough estimate for an AND operation.
            # TODO: Actually run a true AND
            result *= constraint.percent_true(lower_bound, upper_bound)
            if result <= 0:
                return 0
        return result

    def exact_true(self, values):
        for constraint in self.constraints:
            if not constraint(values):
                return False
        return True

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


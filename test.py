import unittest
from ml_constraints.constraints import Label, Constant, LabelOrConstant

# TODO: Add support for 1e8, and 0xa14
GOOD_LABELS = (
    ("4", 4),
    ("0", 0),
    #("1e8", 1e8),
    ("1_000", 1_000),
    #("0xa14", 0xa14),
)
GOOD_INT_CONSTANTS = (
    ("-4", -4.0),
    ("0", 0.0),
    ("1_000", 1_000.0),
    #("1e8", 1.0e8),
    #("-6.2e4", -6.2e4),
    #("0xa14", 0xa14),
    #("-0xd82", -0xd82),
)
GOOD_CONSTANTS = (
    *GOOD_INT_CONSTANTS,
    ("0.4", 0.4),
)
ALWAYS_BAD = (
    (None, TypeError, "Should reject None type"),
    (dict(), TypeError, "Should reject dict type"),
    ("y-1", ValueError, "Should reject negative labels"),
    ("y0.1", ValueError, "Should reject float labels"),
    ("1y", ValueError, "Should reject string with improper type indicators"),
    ("0.1y", ValueError, "Should reject string with improper type indicators"),
    ("f1", ValueError, "Should reject string with improper type indicators"),
    ("f0.1", ValueError, "Should reject string with improper type indicators"),
    ("0", ValueError, "Should reject strings without type indicators"),
    ("0.1", ValueError, "Should reject strings without type indicators"),
)

class TestClasses(unittest.TestCase):
    def test_Label_from_int(self):
        for input_, value in GOOD_LABELS:
            with self.subTest(input_=input_):
                label_from_int = Label(int(input_))
                self.assertIsInstance(label_from_int, Label)
                self.assertEqual(label_from_int, value)
    def test_Label_from_str(self):
        for input_, value in GOOD_LABELS:
            with self.subTest(input_="y" + input_):
                label_from_str = Label("y" + input_)
                self.assertIsInstance(label_from_str, Label)
                self.assertEqual(label_from_str, value)
    def test_Label_errors(self):
        bad_cases = (
            ("4f", ValueError, "Should reject constants"),
            (0.1, TypeError, "Should reject float type"),
            *ALWAYS_BAD
        )
        for input_, error, msg in bad_cases:
            with self.subTest(input_=input_):
                with self.assertRaises(error, msg=msg):
                    label = Label(input_)

    def test_Constant_from_float(self):
        for input_, value in GOOD_CONSTANTS:
            with self.subTest(input_=input_):
                constant_from_float = Constant(float(input_))
                self.assertIsInstance(constant_from_float, Constant)
                self.assertEqual(constant_from_float, value)
    def test_Constant_from_int(self):
        for input_, value in GOOD_INT_CONSTANTS:
            with self.subTest(input_=input_):
                constant_from_int = Constant(int(input_))
                self.assertIsInstance(constant_from_int, Constant)
                self.assertEqual(constant_from_int, value)
    def test_Constant_from_str(self):
        for input_, value in GOOD_CONSTANTS:
            with self.subTest(input_=input_ + "f"):
                constant_from_str = Constant(input_ + "f")
                self.assertIsInstance(constant_from_str, Constant)
                self.assertEqual(constant_from_str, value)
    def test_Constant_errors(self):
        bad_cases = (
            ("2y", ValueError, "Should reject labels"),
            ("0.1y", ValueError, "Should reject labels"),
            *ALWAYS_BAD
        )
        for input_, error, msg in bad_cases:
            with self.subTest(input_=input_):
                with self.assertRaises(error, msg=msg):
                    constant = Constant(input_)

    def test_ConstantOrLabel_Label_from_int(self):
        for input_, value in GOOD_LABELS:
            with self.subTest(input_=input_):
                label_from_int = LabelOrConstant(int(input_))
                self.assertIsInstance(label_from_int, Label)
                self.assertEqual(label_from_int, value)
    def test_ConstantOrLabel_Label_from_str(self):
        for input_, value in GOOD_LABELS:
            with self.subTest(input_="y" + input_):
                label_from_str = LabelOrConstant("y" + input_)
                self.assertIsInstance(label_from_str, Label)
                self.assertEqual(label_from_str, value)
    def test_ConstantOrLabel_Constant_from_float(self):
        for input_, value in GOOD_CONSTANTS:
            with self.subTest(input_=input_):
                constant_from_float = LabelOrConstant(float(input_))
                self.assertIsInstance(constant_from_float, Constant)
                self.assertEqual(constant_from_float, value)
    def test_ConstantOrLabel_Constant_from_str(self):
        for input_, value in GOOD_CONSTANTS:
            with self.subTest(input_=input_ + "f"):
                constant_from_str = LabelOrConstant(input_ + "f")
                self.assertIsInstance(constant_from_str, Constant)
                self.assertEqual(constant_from_str, value)
    def test_ConstantOrLabel_errors(self):
        bad_cases = ALWAYS_BAD
        for input_, error, msg in bad_cases:
            with self.subTest(input_=input_):
                with self.assertRaises(error, msg=msg):
                    label_or_constant = LabelOrConstant(input_)

if __name__ == '__main__':
    unittest.main()

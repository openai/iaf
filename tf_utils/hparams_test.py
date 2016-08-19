import unittest
from hparams import HParams


class HParamsTestCase(unittest.TestCase):
    def test_basic(self):
        hps = HParams(int_value=13, float_value=17.5, bool_value=True, str_value="test")
        self.assertEqual(hps.int_value, 13)
        self.assertEqual(hps.float_value, 17.5)
        self.assertEqual(hps.bool_value, True)
        self.assertEqual(hps.str_value, "test")

    def test_parse(self):
        hps = HParams(int_value=13, float_value=17.5, bool_value=True, str_value="test")
        self.assertEqual(hps.parse("int_value=10").int_value, 10)
        self.assertEqual(hps.parse("float_value=10").float_value, 10)
        self.assertEqual(hps.parse("float_value=10.3").float_value, 10.3)
        self.assertEqual(hps.parse("bool_value=true").bool_value, True)
        self.assertEqual(hps.parse("bool_value=True").bool_value, True)
        self.assertEqual(hps.parse("bool_value=false").bool_value, False)
        self.assertEqual(hps.parse("str_value=value").str_value, "value")

if __name__ == '__main__':
    unittest.main()


import sys
import unittest
from unittest.mock import MagicMock, patch

class TestDepthCalculatorOptimization(unittest.TestCase):
    def setUp(self):
        # Create mocks
        self.mock_numpy = MagicMock()
        class MockNDArray: pass
        self.mock_numpy.ndarray = MockNDArray
        self.mock_numpy.isnan.return_value = False
        self.mock_numpy.std.return_value = 1.0

        self.mock_modules = {
            'numpy': self.mock_numpy,
            'scipy': MagicMock(),
            'scipy.fft': MagicMock(),
            'scipy.signal': MagicMock(),
            'sklearn': MagicMock(),
            'sklearn.cluster': MagicMock(),
            'sklearn.preprocessing': MagicMock(),
            'pyinform': MagicMock(),
            'matplotlib': MagicMock(),
            'matplotlib.pyplot': MagicMock(),
            'networkx': MagicMock(),
            # Intentionally NOT mocking pandas to ensure it is not imported
            # If the code tries to import pandas, it should fail with ModuleNotFoundError
            # (since real pandas is missing in this env)
        }

    def test_calculate_uses_numpy_std_efficiently(self):
        """
        Test that calculate() uses np.std on the sliced array instead of pandas rolling.
        """
        with patch.dict(sys.modules, self.mock_modules):
            # Ensure fresh import
            if 'drr_framework.modules' in sys.modules:
                del sys.modules['drr_framework.modules']

            from drr_framework.modules import DepthCalculator

            calculator = DepthCalculator()

            # Setup data
            data_len = 200
            window_size = 50

            # Create a mock data array
            mock_data = MagicMock()
            # Make it pass isinstance(data, np.ndarray)
            # We need to get the MockNDArray class from our mock_numpy
            mock_data.__class__ = self.mock_numpy.ndarray
            mock_data.__len__.return_value = data_len

            # When sliced, return another mock (the window)
            mock_window = MagicMock()
            mock_data.__getitem__.return_value = mock_window

            # Setup np.std return value specific for this test
            expected_std = 0.567
            self.mock_numpy.std.return_value = expected_std

            # Call the method
            result = calculator.calculate(mock_data, window_size)

            # Verify result
            self.assertEqual(result['resonance_depth'], expected_std)

            # Verify np.std was called with correct arguments
            self.mock_numpy.std.assert_called_once()
            args, kwargs = self.mock_numpy.std.call_args

            # Check first arg is the sliced window
            self.assertIs(args[0], mock_window)

            # Check kwargs has ddof=1
            self.assertEqual(kwargs.get('ddof'), 1)

            # Verify slicing on data was correct: data[-window_size:]
            slice_args = mock_data.__getitem__.call_args[0][0]
            self.assertEqual(slice_args.start, -window_size)
            self.assertIsNone(slice_args.stop)
            self.assertIsNone(slice_args.step)

    def test_calculate_short_data(self):
        """Test behavior when data is shorter than window size"""
        with patch.dict(sys.modules, self.mock_modules):
            if 'drr_framework.modules' in sys.modules:
                del sys.modules['drr_framework.modules']
            from drr_framework.modules import DepthCalculator

            calculator = DepthCalculator()
            window_size = 100
            mock_data = MagicMock()
            mock_data.__class__ = self.mock_numpy.ndarray
            mock_data.__len__.return_value = 50  # Shorter than window

            result = calculator.calculate(mock_data, window_size)

            self.assertEqual(result['resonance_depth'], 0.0)
            # Should not call np.std
            self.mock_numpy.std.assert_not_called()

if __name__ == '__main__':
    unittest.main()

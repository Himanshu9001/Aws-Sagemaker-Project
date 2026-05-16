import sys
import os
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Suppress pandas future warnings for backward compatibility
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas")

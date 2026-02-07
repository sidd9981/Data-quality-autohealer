"""Test to verify setup is correct"""

def test_imports():
    """Test that all dependencies can be imported"""
    import torch
    import pandas as pd
    import numpy as np
    from pyspark.sql import SparkSession
    import kafka
    assert True

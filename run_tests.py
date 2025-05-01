#!/usr/bin/env python3
import unittest
import os
import sys

# Add the parent directory to the path so modules can be imported
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir,'modules'))

# Import test modules
from unittests.test_infection_network import TestInfectionNetwork
from unittests.test_features import TestFeatures
from unittests.test_network_environment_config import TestNetworkEnvironmentConfig
from unittests.test_infection_spread_env import TestInfectionSpreadEnv

def create_test_suite():
    """Create a test suite containing all tests."""
    test_suite = unittest.TestSuite()
    
    # Add test cases from each module
    test_suite.addTest(unittest.makeSuite(TestInfectionNetwork))
    test_suite.addTest(unittest.makeSuite(TestFeatures))
    test_suite.addTest(unittest.makeSuite(TestNetworkEnvironmentConfig))
    test_suite.addTest(unittest.makeSuite(TestInfectionSpreadEnv))
    
    return test_suite

if __name__ == '__main__':
    # Create the test suite
    suite = create_test_suite()
    
    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with status code based on test results
    sys.exit(not result.wasSuccessful())
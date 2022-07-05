class MissingEnvironmentVariableException(Exception):
    """base class for new exception"""
    pass


class InvalidDataRowException(Exception):
    """Raised whenever the contents of a data row is either inaccessible or is too large"""
    pass

class InvalidLabelException(Exception):
    """ Exception for when the data is invalid for vertex."""
    pass


class InvalidDatasetException(Exception):
    """ Exception if the complete dataset doesn't meet vertex requirements"""
    pass
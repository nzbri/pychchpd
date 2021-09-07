from functools import wraps


def return_copy(f=None):
    """
    A decorator to return a copy. This only works with objects that have a 'copy' function.

    Parameters:
        f: function to wrap.

    Returns:
        A copy of the returned object.
    """

    @wraps(f)  # we tell wraps that the function we are wrapping is f
    def create_a_copy(*args, **kwargs):
        value = f(*args, **kwargs)
        return value.copy()

    return create_a_copy

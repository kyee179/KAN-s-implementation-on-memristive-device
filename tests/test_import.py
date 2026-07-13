from kan_memristor import __version__


def test_package_imports():
    assert isinstance(__version__, str)
    assert __version__

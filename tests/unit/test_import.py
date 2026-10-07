"""Smoke test: the package imports, so `pip install -e` and pytest collection work."""


def test_package_imports() -> None:
    import obj_resolution

    assert obj_resolution.__doc__

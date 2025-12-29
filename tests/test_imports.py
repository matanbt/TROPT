
def test_import_tropt():
    import tropt
    assert tropt is not None

def test_import_models():
    from tropt.models import inputs
    assert inputs is not None

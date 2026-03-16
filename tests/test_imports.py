
def test_import_tropt():
    import tropt
    assert tropt is not None

def test_import_models():
    from tropt.model import inputs_manager
    assert inputs_manager is not None

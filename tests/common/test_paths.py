from common import paths


def test_root_is_repo_root():
    # every module resolves data/ and models/ through this one definition,
    # so the nesting depth of src/ is assumed in exactly one place
    assert (paths.ROOT / 'pyproject.toml').is_file()
    assert paths.DATA == paths.ROOT / 'data'
    assert paths.MODELS == paths.ROOT / 'models'

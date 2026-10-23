from students.shared import sft_data
from common.stylometry import targets


def test_sft_data_reexports_targets():
    # moved out of sft_data so the teacher and the evaluation code can use
    # them without importing the students' pipeline
    assert sft_data.select_target_authors is targets.select_target_authors
    assert sft_data.load_profile_rows is targets.load_profile_rows

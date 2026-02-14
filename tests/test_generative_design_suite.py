from drr_framework import GenerativeDesignSuite


def test_generative_suite_architecture_contains_core_engine_blocks():
    suite = GenerativeDesignSuite()
    architecture = suite.architecture()

    assert "mission" in architecture
    assert "core_engine" in architecture

    core_engine = architecture["core_engine"]
    assert "audio_dissection" in core_engine
    assert "resonance_field_mapping" in core_engine
    assert "generative_exploration" in core_engine
    assert "digital_twin_and_export" in core_engine


def test_generative_suite_references_fft_and_mesh_pipeline():
    suite = GenerativeDesignSuite()
    architecture = suite.architecture()

    operations = architecture["core_engine"]["audio_dissection"]["operations"]
    mesh_pipeline = architecture["core_engine"]["resonance_field_mapping"]["mesh_pipeline"]

    assert "short_time_fft" in operations
    assert "extract_isosurface_via_marching_cubes" in mesh_pipeline


def test_generative_suite_closed_loop_ends_with_recalibration():
    flow = GenerativeDesignSuite.closed_loop_flow()
    assert flow[0] == "audio_in_recording"
    assert flow[-1] == "re-ingest_measurements_to_calibrate_the_model"


def test_generative_suite_library_stack_mentions_required_tooling():
    libs = GenerativeDesignSuite.recommended_libraries()

    assert "librosa" in libs["audio_ingest_and_dsp"]
    assert "trimesh" in libs["generative_geometry"]
    assert "blender bpy API" in libs["generative_geometry"]

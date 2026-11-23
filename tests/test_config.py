"""Tests for configuration loading and the class scheme."""

import pytest

from aspilanta.config import ClassScheme, Paths, ProjectConfig, load_config


class TestClassScheme:
    def test_default_scheme(self):
        scheme = ClassScheme()
        assert scheme.n_base == 3
        assert scheme.n_full == 4
        assert scheme.full[scheme.target_id] == "Aspilanta"

    def test_synthetic_map_targets_only_the_target_class(self):
        scheme = ClassScheme()
        assert scheme.synthetic_class_map == {0: 3}

    def test_distractors_are_dropped_not_mapped(self):
        """The generator's distractor id must never appear in the class map.

        If it did, it would become a real class - the Section 11.1 defect.
        """
        scheme = ClassScheme()
        assert scheme.generator_distractor_id in scheme.synthetic_drop_ids
        assert scheme.generator_distractor_id not in scheme.synthetic_class_map


class TestPaths:
    def test_derived_paths_hang_off_work(self, tmp_path):
        paths = Paths(work=tmp_path, inputs=tmp_path / "in")
        assert paths.exp3_dataset == tmp_path / "exp3_dataset"
        assert paths.runs == tmp_path / "runs" / "detect"
        assert paths.results == tmp_path / "thesis_results"

    def test_ensure_creates_writable_dirs(self, tmp_path):
        paths = Paths(work=tmp_path / "w", inputs=tmp_path / "i").ensure()
        assert paths.results.exists()
        assert paths.cutouts_aspilanta.exists()

    def test_strings_are_coerced_to_paths(self, tmp_path):
        paths = Paths(work=str(tmp_path), inputs=str(tmp_path))
        assert paths.work.is_absolute() or paths.work == tmp_path


class TestLoadConfig:
    def test_defaults_without_a_file(self):
        config = load_config()
        assert isinstance(config, ProjectConfig)
        assert config.solution.freeze == 0        # the proposed setting
        assert config.deployment.use_tiling is False
        assert config.deployment.use_clip_filter is False

    def test_yaml_overlay_is_partial(self, tmp_path):
        path = tmp_path / "override.yaml"
        path.write_text("solution:\n  epochs: 5\n  freeze: 10\n")
        config = load_config(path)
        assert config.solution.epochs == 5
        assert config.solution.freeze == 10
        assert config.solution.copy_paste == 0.3   # untouched default survives

    def test_unknown_key_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("solution:\n  not_a_real_option: 1\n")
        with pytest.raises(KeyError):
            load_config(path)

    def test_unknown_section_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("nonsense:\n  a: 1\n")
        with pytest.raises(KeyError):
            load_config(path)

    def test_ultralytics_kwargs_honour_overrides(self):
        config = load_config()
        kwargs = config.solution.ultralytics_kwargs(freeze=10, copy_paste=0.0)
        assert kwargs["freeze"] == 10
        assert kwargs["copy_paste"] == 0.0
        # The base config must not be mutated by an override.
        assert config.solution.freeze == 0

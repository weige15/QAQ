import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STRUCTURE = json.loads((ROOT / "docs/model_structure.json").read_text())


class ModelInspectionEvidenceTest(unittest.TestCase):
    def test_identity_and_dimensions_are_pinned(self):
        identity = STRUCTURE["identity"]
        dimensions = STRUCTURE["global_dimensions"]
        self.assertEqual(identity["repository"], "Qwen/Qwen3-4B")
        self.assertEqual(identity["revision"], "1cfa9a7208912126459214e8b04321603b3df60c")
        self.assertEqual(identity["architecture_class"], "Qwen3ForCausalLM")
        self.assertEqual(dimensions["num_decoder_layers"], 36)
        self.assertEqual(dimensions["hidden_size"], 2560)
        self.assertEqual(dimensions["intermediate_ffn_size"], 9728)

    def test_projection_paths_and_counts_are_complete(self):
        targets = STRUCTURE["quantization_targets"]
        self.assertEqual(targets["attention_projection_count"], 144)
        self.assertEqual(targets["ffn_projection_count"], 108)
        self.assertEqual(targets["total_count"], 252)
        self.assertTrue(targets["no_duplicates"])
        self.assertEqual(len(targets["module_names"]), 252)
        self.assertEqual(
            targets["module_names"][0], "model.layers.0.self_attn.q_proj"
        )
        self.assertEqual(
            targets["module_names"][-1], "model.layers.35.mlp.down_proj"
        )
        self.assertEqual(
            {name.rsplit(".", 1)[-1] for name in targets["module_names"]},
            {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
        )

    def test_non_targets_are_excluded(self):
        names = set(STRUCTURE["quantization_targets"]["module_names"])
        non_targets = {item["path"] for item in STRUCTURE["non_target_components"]}
        self.assertFalse(any("embed_tokens" in name for name in names))
        self.assertFalse(any("lm_head" in name for name in names))
        self.assertFalse(any("norm" in name for name in names))
        self.assertFalse(any(name in non_targets for name in names))
        self.assertTrue(STRUCTURE["tied_weights"]["config_tie_word_embeddings"])

    def test_source_and_backend_checks_pass_without_runtime_model(self):
        markers = STRUCTURE["source_markers"]
        backend = STRUCTURE["any_precision_analysis"]
        self.assertTrue(markers["required_modeling_classes_present"])
        self.assertTrue(markers["configuration_class_present"])
        self.assertTrue(markers["attention_has_qk_norm"])
        self.assertTrue(markers["causal_lm_declares_tied_head"])
        self.assertFalse(backend["qwen3_explicitly_supported"])
        self.assertTrue(backend["structural_linear_fit"]["explicit_mapping_unambiguous"])
        self.assertFalse(STRUCTURE["weight_policy"]["weights_loaded"])
        self.assertFalse(STRUCTURE["weight_policy"]["weight_shards_downloaded"])


if __name__ == "__main__":
    unittest.main()

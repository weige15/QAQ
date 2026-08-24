import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACTUAL = json.loads((ROOT / "docs/actual_model_modules.json").read_text())


class ActualModelInspectionEvidenceTest(unittest.TestCase):
    def test_identity_and_load_are_pinned(self):
        self.assertEqual(ACTUAL["exact_model_repository"], "Qwen/Qwen3-4B")
        self.assertEqual(
            ACTUAL["exact_model_revision"],
            "1cfa9a7208912126459214e8b04321603b3df60c",
        )
        self.assertEqual(ACTUAL["model_class"].split(".")[-1], "Qwen3ForCausalLM")
        self.assertEqual(ACTUAL["dtype"], "torch.bfloat16")
        self.assertEqual(ACTUAL["device_placement"], ["cuda:3"])
        self.assertFalse(ACTUAL["quantization_performed"])

    def test_actual_targets_have_complete_counts_and_unique_paths(self):
        targets = ACTUAL["target_modules"]
        paths = [target["full_module_path"] for target in targets]
        self.assertEqual(ACTUAL["attention_target_count"], 144)
        self.assertEqual(ACTUAL["ffn_target_count"], 108)
        self.assertEqual(ACTUAL["total_target_count"], 252)
        self.assertEqual(ACTUAL["total_target_count"], len(targets))
        self.assertEqual(len(paths), len(set(paths)))
        self.assertFalse(ACTUAL["target_paths_duplicate"])
        self.assertEqual(ACTUAL["layer_indices"], list(range(36)))

    def test_actual_target_properties_match_model_mapping(self):
        comparison = ACTUAL["s00_mapping_comparison"]
        self.assertEqual(comparison["result"], "MATCH")
        self.assertEqual(comparison["missing_expected_targets"], [])
        self.assertEqual(comparison["unexpected_target_paths"], [])
        self.assertEqual(comparison["module_property_mismatches"], [])
        self.assertTrue(comparison["all_targets_supported_linear"])
        self.assertTrue(comparison["dimensions_and_biases_match"])
        for target in ACTUAL["target_modules"]:
            self.assertEqual(target["python_module_class"], "torch.nn.modules.linear.Linear")
            self.assertTrue(target["proposed_quantization_target"])
            self.assertFalse(target["bias_present"])

    def test_exclusions_and_unexpected_linears_are_recorded(self):
        exclusions = ACTUAL["excluded_module_categories"]
        for category in (
            "token_embeddings",
            "lm_output_head",
            "final_normalization",
            "per_layer_normalization",
            "qk_normalization",
            "rotary_position",
            "activation_functions",
            "kv_cache",
        ):
            self.assertIn(category, exclusions)
            self.assertEqual(
                exclusions[category]["policy"],
                "excluded" if category != "kv_cache" else "runtime structure, not a model module",
            )
        unexpected = ACTUAL["unexpected_linear_modules"]
        self.assertEqual([item["path"] for item in unexpected], ["lm_head"])
        self.assertEqual(unexpected[0]["classification"], "excluded output head")

    def test_tied_weights_and_smoke_forward(self):
        tied = ACTUAL["tied_weight_verification"]
        self.assertTrue(tied["config_tie_word_embeddings"])
        self.assertTrue(tied["same_parameter_object"])
        self.assertTrue(tied["same_storage_pointer"])
        smoke = ACTUAL["full_precision_smoke_forward"]
        self.assertTrue(smoke["finite_values"])
        self.assertEqual(smoke["logits_shape"][0], 1)
        self.assertEqual(smoke["logits_shape"][2], 151936)


if __name__ == "__main__":
    unittest.main()

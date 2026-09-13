"""
Unit tests for the Disagreement Principle and Representation Scope Registry.

Verifies non-erasure scope resolution, multi-scale divergence detection, template formatting,
and intellectual humility metadata handling using archetype profiles for Samira (macroprudential)
and Christopher (local distribution-sensitive tracker).
"""

import unittest
from drr_framework.disagreement import (
    DRR_ScopeResolver,
    ObservationalPerspective,
    ObservationalScale,
)


class TestDisagreementPrinciple(unittest.TestCase):
    """Test suite for ObservationalPerspective and DRR_ScopeResolver."""

    def setUp(self) -> None:
        """Set up mock archetypes for Samira (Macro) and Christopher (Local)."""
        # Samira's profile archetype: Fed Macroprudential Model (High confidence MACRO)
        self.samira_macro_archetype = ObservationalPerspective(
            source_id="Fed_Macroprudential_Model",
            scale="MACRO",
            confidence_score=0.95,
            indicators={
                "financial_system": "stabilizing",
                "inflation_pressure": "moderating",
            },
            evidence_provenance="fed_vintage_hash_2025_q1_a8f3c",
            dissent_logged=False,
        )

        # Christopher's profile archetype: Distribution-sensitive Local Tracker (LOCAL)
        self.christopher_local_archetype = ObservationalPerspective(
            source_id="Local_Material_Survey",
            scale=ObservationalScale.LOCAL,
            confidence_score=0.82,
            indicators={
                "purchasing_power": "deteriorating",
                "housing_affordability": "deteriorating",
            },
            evidence_provenance="survey_vintage_hash_2025_q1_c92b1",
            dissent_logged=True,
        )

    def test_non_erasure_invariant_preserves_both_perspectives(self) -> None:
        """Verify high confidence MACRO perspective does NOT overwrite or erase LOCAL perspective."""
        resolver = DRR_ScopeResolver()
        resolver.register_perspective(self.samira_macro_archetype)
        resolver.register_perspective(self.christopher_local_archetype)

        registered = resolver.perspectives
        self.assertEqual(len(registered), 2)

        # Confirm exact contents survive in full fidelity
        source_ids = [p.source_id for p in registered]
        self.assertIn("Fed_Macroprudential_Model", source_ids)
        self.assertIn("Local_Material_Survey", source_ids)

        samira_p = next(p for p in registered if p.source_id == "Fed_Macroprudential_Model")
        christopher_p = next(p for p in registered if p.source_id == "Local_Material_Survey")

        self.assertEqual(samira_p.confidence_score, 0.95)
        self.assertEqual(samira_p.scale, ObservationalScale.MACRO)
        self.assertEqual(christopher_p.confidence_score, 0.82)
        self.assertEqual(christopher_p.scale, ObservationalScale.LOCAL)
        self.assertTrue(christopher_p.dissent_logged)

    def test_divergence_synthesis_engine_output_and_template(self) -> None:
        """Verify generate_drr_conclusion outputs required natural language template and humility metadata."""
        resolver = DRR_ScopeResolver([
            self.samira_macro_archetype,
            self.christopher_local_archetype,
        ])

        conclusion_payload = resolver.generate_drr_conclusion()

        self.assertTrue(conclusion_payload["has_divergence"])
        self.assertEqual(conclusion_payload["status"], "divergent_scopes_preserved")
        self.assertEqual(conclusion_payload["perspectives_evaluated"], 2)

        # Verify natural language template structure
        conclusion_text = conclusion_payload["conclusion"]
        expected_template_suffix = (
            "These findings operate at different observational scopes and should "
            "be interpreted together rather than collapsed into a single state."
        )
        self.assertIn("Aggregate financial-system indicators support", conclusion_text)
        self.assertIn(", while material indicators for the evaluated population support", conclusion_text)
        self.assertIn(expected_template_suffix, conclusion_text)

        # Check macro and local state extraction
        self.assertIn("stabilizing", conclusion_payload["macro_state"])
        self.assertIn("deteriorating", conclusion_payload["local_state"])

        # Check intellectual humility metadata block
        humility = conclusion_payload["intellectual_humility"]
        self.assertIn("accuracy_within_representation", humility)
        self.assertIn("completeness_of_representation", humility)
        self.assertTrue(humility["non_erasure_invariant_maintained"])
        self.assertIn("MACRO avg confidence: 0.95", humility["accuracy_within_representation"])
        self.assertIn("LOCAL/MICRO avg confidence: 0.82", humility["accuracy_within_representation"])

    def test_concordant_perspectives_returns_consensus(self) -> None:
        """Verify concordant perspectives across scales synthesize without divergence flag."""
        macro_stable = ObservationalPerspective(
            source_id="Macro_Model",
            scale="MACRO",
            confidence_score=0.90,
            indicators={"economy": "stabilizing"},
            evidence_provenance="prov_01",
        )
        local_stable = ObservationalPerspective(
            source_id="Local_Model",
            scale="LOCAL",
            confidence_score=0.88,
            indicators={"local_economy": "stabilizing"},
            evidence_provenance="prov_02",
        )

        resolver = DRR_ScopeResolver([macro_stable, local_stable])
        payload = resolver.generate_drr_conclusion()

        self.assertFalse(payload["has_divergence"])
        self.assertEqual(payload["status"], "concordant_consensus")
        self.assertIn("concordance", payload["conclusion"])

    def test_observational_perspective_field_validation(self) -> None:
        """Verify validation logic for confidence_score and scale."""
        # Invalid confidence_score > 1.0
        with self.assertRaises(ValueError):
            ObservationalPerspective(
                source_id="Invalid_Conf",
                scale="MACRO",
                confidence_score=1.5,
                indicators={},
                evidence_provenance="test",
            )

        # Invalid confidence_score < 0.0
        with self.assertRaises(ValueError):
            ObservationalPerspective(
                source_id="Invalid_Conf",
                scale="MACRO",
                confidence_score=-0.1,
                indicators={},
                evidence_provenance="test",
            )

        # Invalid scale string
        with self.assertRaises(ValueError):
            ObservationalPerspective(
                source_id="Invalid_Scale",
                scale="GLOBAL_SUPER_SCALE",
                confidence_score=0.8,
                indicators={},
                evidence_provenance="test",
            )


if __name__ == "__main__":
    unittest.main()

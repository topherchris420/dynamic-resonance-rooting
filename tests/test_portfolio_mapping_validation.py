"""Tests for automated portfolio schema mapping, crosswalk resolution, and data validation."""

import os
import tempfile
import unittest
import pandas as pd

from drr_framework.supervisory import (
    AutomatedSchemaMapper,
    FilingContext,
    MDRMCrosswalk,
    MDRMCrosswalkRule,
    PortfolioValidationReport,
    SemanticRegistry,
    VerificationStatus,
    build_portfolio_registry,
    bundled_registry,
    ingest_wide_filing,
    validate_portfolio_dataset,
)


class TestPortfolioMappingValidation(unittest.TestCase):
    def setUp(self):
        self.base_registry = bundled_registry()
        self.reporting_period = "2026-03-31"
        self.available_as_of = "2026-09-12T14:00:00Z"
        self.form = "FR Y-9C"

    def test_mdrm_crosswalk(self):
        crosswalk = MDRMCrosswalk()
        # Test default mappings
        self.assertEqual(
            crosswalk.translate("FR Y-9C", "BHCK2170", "FFIEC 031"), "RCFD2170"
        )
        self.assertEqual(
            crosswalk.translate("FR Y-9C", "BHCK2170", "FFIEC 041"), "RCON2170"
        )
        # Test unmapped query
        self.assertIsNone(crosswalk.translate("FR Y-9C", "BHCK9999", "FFIEC 031"))

        # Test custom rule addition
        custom_rule = MDRMCrosswalkRule("FR Y-9C", "BHCK9999", "FFIEC 031", "RCFD9999")
        crosswalk.add_rule(custom_rule)
        self.assertEqual(
            crosswalk.translate("FR Y-9C", "BHCK9999", "FFIEC 031"), "RCFD9999"
        )

    def test_automated_schema_mapper(self):
        mapper = AutomatedSchemaMapper()

        # Generate single metric definition
        metric_def = mapper.generate_definition(
            code="BHCK0390",
            form=self.form,
            reporting_period=self.reporting_period,
            label="INVESTMENT SECURITIES",
        )
        self.assertEqual(metric_def.code, "BHCK0390")
        self.assertEqual(metric_def.form, self.form)
        self.assertEqual(metric_def.status, VerificationStatus.VERIFIED)

        # Expand existing registry
        new_registry = mapper.expand_registry(
            existing_registry=self.base_registry,
            new_metrics=["BHCK0390", "BHCK1400"],
            form=self.form,
            reporting_period=self.reporting_period,
        )
        self.assertGreater(len(new_registry.definitions), len(self.base_registry.definitions))

        # Resolve newly expanded definition
        resolved = new_registry.resolve(
            form=self.form,
            code="BHCK0390",
            period=self.reporting_period,
            as_of=self.available_as_of,
            allow_synthetic=True,
        )
        self.assertEqual(resolved.code, "BHCK0390")

    def test_validate_portfolio_dataset_valid(self):
        df = pd.DataFrame({
            "RSSD_ID": ["1073757", "1073758"],
            "LEGAL_NAME": ["Bank Alpha", "Bank Beta"],
            "BHCK0081": [100.0, 200.0],
            "BHCK2170": [5000.0, 10000.0],
        })

        report = validate_portfolio_dataset(
            frame_or_path=df,
            registry=self.base_registry,
            form=self.form,
            reporting_period=self.reporting_period,
            available_as_of=self.available_as_of,
        )

        self.assertTrue(report.is_valid)
        self.assertEqual(report.total_institutions, 2)
        self.assertEqual(report.total_metrics, 2)
        self.assertEqual(len(report.unmapped_metrics), 0)
        self.assertIn("PASSED", report.summary())

    def test_validate_portfolio_dataset_unmapped_and_formatting_errors(self):
        df = pd.DataFrame({
            "RSSD_ID": ["1073757.0", "1073758"],  # float RSSD_ID error
            "LEGAL_NAME": ["Bank Alpha", "Bank Beta"],
            "BHCK0081": [100.0, 200.0],
            "BHCK9991": [10.0, 20.0],  # unmapped 8-char MDRM
            "ZERO_VAR_METRIC": [50.0, 50.0],  # zero variance non-MDRM
        })

        report = validate_portfolio_dataset(
            frame_or_path=df,
            registry=self.base_registry,
            form=self.form,
            reporting_period=self.reporting_period,
            available_as_of=self.available_as_of,
            definition_version="FRY9C-2026-03-verified-2026-09-12",
            auto_generate_patches=True,
        )

        self.assertFalse(report.is_valid)
        self.assertEqual(len(report.id_formatting_errors), 1)
        self.assertIn("BHCK9991", report.unmapped_metrics)
        self.assertIn("ZERO_VAR_METRIC", report.unmapped_metrics)
        self.assertIn("ZERO_VAR_METRIC", report.zero_variance_metrics)
        self.assertIsNotNone(report.auto_patch_registry)

        # Confirm patch registry resolves unmapped metrics
        patched_registry = report.auto_patch_registry
        resolved_unmapped = patched_registry.resolve(
            form=self.form,
            code="BHCK9991",
            period=self.reporting_period,
            as_of=self.available_as_of,
            allow_synthetic=True,
        )
        self.assertEqual(resolved_unmapped.code, "BHCK9991")

    def test_end_to_end_ingestion_with_auto_patch(self):
        df = pd.DataFrame({
            "RSSD_ID": ["1073757", "1073758"],
            "LEGAL_NAME": ["Bank Alpha", "Bank Beta"],
            "BHCK0081": [100.0, 200.0],
            "BHCK9992": [15.0, 25.0],
        })

        report = validate_portfolio_dataset(
            frame_or_path=df,
            registry=self.base_registry,
            form=self.form,
            reporting_period=self.reporting_period,
            available_as_of=self.available_as_of,
            definition_version="FRY9C-2026-03-verified-2026-09-12",
            auto_generate_patches=True,
        )

        self.assertIsNotNone(report.auto_patch_registry)

        # Perform atomic ingestion using auto-patched registry
        context = FilingContext(
            form=self.form,
            reporting_period=self.reporting_period,
            original_filing_date="2026-05-01T00:00:00Z",
            ingestion_date="2026-09-12T14:00:00Z",
            available_as_of=self.available_as_of,
            source_vintage="2026Q1-download-1",
            source="https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload",
            source_hash="f80f4d4734a67976906f8249f9e9ed713b8336085564914fa61f04aaff4e253a",
            definition_version="FRY9C-2026-03-verified-2026-09-12",
        )

        store = ingest_wide_filing(
            df,
            report.auto_patch_registry,
            context,
            metric_columns=["BHCK0081", "BHCK9992"],
        )

        self.assertEqual(len(store.observations), 4)

    def test_build_portfolio_registry_helper(self):
        portfolio_reg = build_portfolio_registry(
            extra_definitions=["EXTRA_1", "EXTRA_2"],
            default_form=self.form,
            default_reporting_period=self.reporting_period,
        )
        self.assertGreater(len(portfolio_reg.definitions), len(self.base_registry.definitions))


if __name__ == "__main__":
    unittest.main()

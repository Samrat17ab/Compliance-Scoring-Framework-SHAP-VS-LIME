"""Independent checks of the shipped research evidence; no full training dataset needed."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

import lightgbm as lgb
import numpy as np
from PIL import Image
import pymupdf
from scipy.special import expit
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/paper_c3b_c6_evidence'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


class EvidenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = read_json(OUT / 'evidence.json')
        cls.rows = cls.data['per_instance']
        with np.load(OUT / 'full_shap_and_inputs.npz', allow_pickle=False) as arrays:
            cls.arrays = {key: arrays[key].copy() for key in arrays.files}
        cls.model = lgb.Booster(model_file=str(OUT / 'lightgbm_model.txt'))

    def test_artifact_checksums(self):
        manifest = read_json(ROOT / 'verification/artifact_manifest.json')
        self.assertGreater(len(manifest['files']), 30)
        for entry in manifest['files']:
            with self.subTest(path=entry['path']):
                raw = (ROOT / entry['path']).read_bytes()
                self.assertEqual(len(raw), entry['bytes'])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), entry['sha256'])

    def test_scientific_source_preservation_and_syntax(self):
        manifest = read_json(ROOT / 'verification/source_manifest.json')
        for name, expected in manifest.items():
            with self.subTest(source=name):
                source = (ROOT / name).read_text(encoding='utf-8')
                ast.parse(source, filename=name)
                self.assertEqual(hashlib.sha256(source.encode('utf-8')).hexdigest(), expected)

    def test_selected_manuscript(self):
        with pymupdf.open(ROOT / 'paper/Evaluating_SHAP_and_LIME_Against_EU_AI_Act_A.pdf') as doc:
            self.assertEqual(len(doc), 6)
            text = '\n'.join(page.get_text() for page in doc)
            for phrase in ['Compliance Scoring Framework', 'Samrat Lamsal', 'Sumana De', 'Sulav Poudel', '4.12', '2.50']:
                self.assertIn(phrase, text)
            self.assertTrue(all(len(page.get_text()) > 300 for page in doc))

    def test_sample_identity_dimensions_and_model_hash(self):
        self.assertEqual(len(self.rows), 1000)
        ids = [row['applicant_id'] for row in self.rows]
        self.assertEqual(len(set(ids)), 1000)
        self.assertEqual(ids, self.data['sample_applicant_ids'])
        self.assertEqual([r['sample_position'] for r in self.rows], list(range(1000)))
        self.assertEqual(self.arrays['inputs'].shape, (1000, 120))
        self.assertEqual(self.arrays['shap_values'].shape, (1000, 120))
        self.assertTrue(np.isfinite(self.arrays['inputs']).all())
        digest = hashlib.sha256((OUT / 'lightgbm_model.txt').read_bytes()).hexdigest()
        self.assertEqual(digest, self.data['configuration']['model_sha256'])
        self.assertEqual(self.model.current_iteration(), 190)

    def test_saved_model_predictions_and_shap_reconstruction(self):
        inputs = self.arrays['inputs']
        margins = self.model.predict(inputs, raw_score=True, num_threads=2)
        scores = self.model.predict(inputs, num_threads=2)
        np.testing.assert_allclose(margins, [r['model_log_odds'] for r in self.rows], rtol=0, atol=1e-10)
        np.testing.assert_allclose(scores, [r['score'] for r in self.rows], rtol=0, atol=1e-12)
        np.testing.assert_allclose(scores, expit(margins), rtol=0, atol=1e-12)
        reconstruction = np.array([r['shap_base'] for r in self.rows]) + self.arrays['shap_values'].sum(axis=1)
        np.testing.assert_allclose(reconstruction, margins, rtol=0, atol=1e-10)
        errors = np.abs(reconstruction - np.array([r['model_log_odds'] for r in self.rows]))
        self.assertAlmostEqual(errors.max(), self.data['part_a']['traceability']['shap_reconstruction_error_max'], delta=1e-14)

    def test_shap_top_ten_matches_full_attribution_array(self):
        for index, row in enumerate(self.rows):
            terms = row['shap_top10']
            expected_indices = set(np.argsort(-np.abs(self.arrays['shap_values'][index]))[:10].tolist())
            self.assertEqual({t['feature_index'] for t in terms}, expected_indices)
            for term in terms:
                j = term['feature_index']
                self.assertAlmostEqual(term['phi'], self.arrays['shap_values'][index, j], places=12)
                self.assertAlmostEqual(term['value'], self.arrays['inputs'][index, j], places=10)

    def test_lime_gap_aggregates_and_representation(self):
        gaps = np.array([abs(r['lime']['surrogate_prediction'] - r['score']) for r in self.rows])
        stats = self.data['part_a']['traceability']
        for name, value in [('mean', gaps.mean()), ('median', np.median(gaps)), ('p90', np.percentile(gaps, 90))]:
            self.assertAlmostEqual(value, stats['lime_absolute_surrogate_gap_' + name], places=12)
        count = sum('<' in t['condition'] or '>' in t['condition'] for r in self.rows for t in r['lime']['top10'])
        self.assertEqual(count, self.data['part_a']['representation']['lime_bin_or_range_item_count'])
        self.assertEqual(count, 10000)

    def test_explanation_composition_and_gender_counts(self):
        part = self.data['part_b']
        for method in ['shap', 'lime']:
            counts = dict.fromkeys(part['top10_composition'][method], 0)
            burden = gender = 0
            for row in self.rows:
                terms = row['shap_top10'] if method == 'shap' else row['lime']['top10']
                self.assertEqual(len({t['feature'] for t in terms}), 10)
                for term in terms:
                    self.assertEqual(term['category'], part['feature_category_map'][term['feature']])
                    counts[term['category']] += 1
                burden += sum(t['category'] in ['document_flags', 'encoded_categoricals'] for t in terms) >= 5
                gender += any(t['feature'] == 'CODE_GENDER' for t in terms)
            self.assertEqual(sum(counts.values()), 10000)
            for category, count in counts.items():
                saved = part['top10_composition'][method][category]
                self.assertEqual(count, saved['total_items'])
                self.assertAlmostEqual(count / 100, saved['mean_pct_of_top10'], places=10)
            self.assertEqual(burden, part['at_least_5_document_flags_or_encoded_categoricals'][method]['instance_count'])
            self.assertEqual(gender, part['code_gender_in_top10'][method]['instance_count'])

    def test_example_selection_matches_prespecified_rule(self):
        for decision in ['accepted', 'rejected']:
            group = [r for r in self.rows if (r['score'] < .5) == (decision == 'accepted')]
            median = float(np.median([r['lime']['r2'] for r in group]))
            selected = min(group, key=lambda r: (abs(r['lime']['r2'] - median), r['sample_position']))
            expected = self.data['part_c']['selected'][decision]
            self.assertEqual(selected['applicant_id'], expected['applicant_id'])
            self.assertEqual(len(group), expected['group_count'])
            self.assertAlmostEqual(median, expected['group_median_lime_r2'], places=12)

    def test_corrected_paired_wilcoxon(self):
        data = read_json(ROOT / 'supplemental_C_paired_jaccard.json')
        s, l = np.array(data['shap_per_instance']), np.array(data['lime_per_instance'])
        self.assertEqual(len(s), data['n_instances'])
        self.assertEqual(len(l), len(s))
        result = wilcoxon(s, l)
        self.assertEqual(result.statistic, data['wilcoxon_stat'])
        np.testing.assert_allclose(result.pvalue, data['wilcoxon_p'], rtol=1e-10, atol=0)
        self.assertAlmostEqual(l.mean(), data['aggregate_lime_mean'], places=12)
        self.assertTrue(np.all(s == 1))
        self.assertTrue(np.all(l < 1))

    def test_corrected_scores_and_monte_carlo(self):
        data = read_json(ROOT / 'supplemental_D_corrected_scores.json')
        s, l = np.array(list(data['shap_scores'].values())), np.array(list(data['lime_scores'].values()))
        self.assertEqual(len(s), 8)
        self.assertEqual(len(l), 8)
        self.assertEqual(s.mean(), data['shap_overall'])
        self.assertEqual(l.mean(), data['lime_overall'])
        weights = np.random.RandomState(42).dirichlet(np.ones(8), size=data['monte_carlo']['n_sim'])
        wins = int(np.count_nonzero(weights @ s > weights @ l))
        self.assertEqual(wins, data['monte_carlo']['shap_wins'])
        self.assertAlmostEqual(wins / len(weights), data['monte_carlo']['robustness_ratio'], places=12)
        a = read_json(ROOT / 'supplemental_A_c4_probability.json')
        self.assertAlmostEqual(a['exact_interventional_prob_gap'], data['c4_shap_prob_gap'], delta=5e-7)

    def test_summary_figures_have_expected_dimensions(self):
        for name in ['accepted_example.png', 'rejected_example.png', 'summary_1000.png']:
            with Image.open(OUT / name) as im:
                self.assertEqual(im.size, (3200, 1800))
                im.verify()


if __name__ == '__main__':
    unittest.main()

from unittest.mock import patch

from django.test import TestCase
from PIL import Image

from .views import predict_image


class VerificationApiCORSOptionsTests(TestCase):
    def test_verify_endpoint_accepts_preflight_options(self):
        response = self.client.options('/verify/')

        self.assertEqual(response.status_code, 200)
        self.assertIn('Access-Control-Allow-Methods', response.headers)
        self.assertIn('POST', response.headers['Access-Control-Allow-Methods'])


class VerificationApiInferenceTests(TestCase):
    def test_predict_image_uses_local_inference_when_available(self):
        pil_img = Image.new('RGB', (224, 224), color='red')

        with patch('verification.views.run_local_inference', return_value={
            'fake_probability': 0.82,
            'real_probability': 0.18,
            'model_source': 'local',
        }) as mock_run_local_inference:
            result = predict_image(pil_img)

        self.assertEqual(result['model_source'], 'local')
        self.assertEqual(result['fake_probability'], 0.82)
        self.assertEqual(result['real_probability'], 0.18)
        self.assertIn('artifact_components', result)
        self.assertIn('artifact_score', result)
        self.assertIn('artifact_evidence', result)
        self.assertEqual(result['artifact_components']['face_integrity'], 0.0)
        self.assertEqual(result['artifact_components']['gan_artifact'], 0.0)
        self.assertIn('texture_coh', result['artifact_components'])
        self.assertIn('artifact_evidence', result)
        mock_run_local_inference.assert_called_once_with(pil_img)

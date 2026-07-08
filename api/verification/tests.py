from django.test import TestCase


class VerificationApiCORSOptionsTests(TestCase):
    def test_verify_endpoint_accepts_preflight_options(self):
        response = self.client.options('/verify/')

        self.assertEqual(response.status_code, 200)
        self.assertIn('Access-Control-Allow-Methods', response.headers)
        self.assertIn('POST', response.headers['Access-Control-Allow-Methods'])

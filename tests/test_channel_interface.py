import os
import re
import unittest
from unittest.mock import patch

import app as ghost


class ChannelInterfaceTests(unittest.TestCase):
    def test_dashboard_and_demo_render_without_conflict_markers(self):
        environment={key:'' for key in ('GHOST_ADMIN_PASSWORD','GHOST_ADMIN_PASSWORD_HASH',
            'WHATSAPP_APP_SECRET','WHATSAPP_PHONE_NUMBER_ID','GHOST_GMAIL_ACCOUNT','GHOST_GMAIL_APP_PASSWORD')}
        with patch.dict(os.environ,environment):
            for path in ('/dashboard','/demo'):
                with self.subTest(path=path):
                    response=ghost.app.test_client().get(path)
                    self.assertEqual(response.status_code,200)
                    html=response.get_data(as_text=True)
                    self.assertNotRegex(html,re.compile(r'^(?:<{7}|={7}|>{7})',re.M))
                    self.assertNotIn('Updated upstream',html)
                    self.assertNotIn('Stashed changes',html)
                    self.assertIn('<title>Mteja-Insight',html)
                    self.assertIn('data-connect-channel="choose"',html)
                    self.assertIn('id="provider-connections"',html)
                    self.assertIn('id="connection-dialog"',html)

    def test_signed_meta_callback_has_one_handler_and_legacy_route_is_separate(self):
        callbacks=[rule for rule in ghost.app.url_map.iter_rules() if rule.rule=='/webhooks/whatsapp']
        self.assertEqual(len(callbacks),1)
        self.assertEqual(callbacks[0].endpoint,'meta_whatsapp')
        self.assertTrue(any(rule.rule=='/webhook/whatsapp' for rule in ghost.app.url_map.iter_rules()))


if __name__=='__main__':unittest.main()

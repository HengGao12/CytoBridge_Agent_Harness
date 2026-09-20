import unittest

from langchain_core.messages import SystemMessage

from cytobridge_agent.utils.codex_chat import ChatCodexOAuth


class ChatCodexOAuthTests(unittest.TestCase):
    def test_system_only_prompt_gets_neutral_user_input(self) -> None:
        llm = ChatCodexOAuth()
        formatted = llm._format_messages([SystemMessage(content="compacted context")])

        self.assertEqual(len(formatted), 1)
        self.assertEqual(formatted[0]["role"], "user")
        self.assertIn("Continue", formatted[0]["content"])


if __name__ == "__main__":
    unittest.main()

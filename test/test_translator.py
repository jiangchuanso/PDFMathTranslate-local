import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from textwrap import dedent
from unittest import mock

from ollama import ResponseError as OllamaResponseError

from pdf2zh import cache
from pdf2zh import translator as translator_module
from pdf2zh.config import ConfigManager
from pdf2zh.translator import BaseTranslator, OllamaTranslator, OpenAIlikedTranslator

# Since it is necessary to test whether the functionality meets the expected requirements,
# private functions and private methods are allowed to be called.
# pyright: reportPrivateUsage=false


class AutoIncreaseTranslator(BaseTranslator):
    name = "auto_increase"
    n = 0

    def do_translate(self, text):
        self.n += 1
        return str(self.n)


class TestTranslator(unittest.TestCase):
    def setUp(self):
        self.test_db = cache.init_test_db()

    def tearDown(self):
        cache.clean_test_db(self.test_db)

    def test_cache(self):
        translator = AutoIncreaseTranslator("en", "zh", "test", False)
        # First translation should be cached
        text = "Hello World"
        first_result = translator.translate(text)

        # Second translation should return the same result from cache
        second_result = translator.translate(text)
        self.assertEqual(first_result, second_result)

        # Different input should give different result
        different_text = "Different Text"
        different_result = translator.translate(different_text)
        self.assertNotEqual(first_result, different_result)

        # Test cache with ignore_cache=True
        translator.ignore_cache = True
        no_cache_result = translator.translate(text)
        self.assertNotEqual(first_result, no_cache_result)

    def test_add_cache_impact_parameters(self):
        translator = AutoIncreaseTranslator("en", "zh", "test", False)

        # Test cache with added parameters
        text = "Hello World"
        first_result = translator.translate(text)
        translator.add_cache_impact_parameters("test", "value")
        second_result = translator.translate(text)
        self.assertNotEqual(first_result, second_result)

        # Test cache with ignore_cache=True
        no_cache_result1 = translator.translate(text, ignore_cache=True)
        self.assertNotEqual(first_result, no_cache_result1)

        translator.ignore_cache = True
        no_cache_result2 = translator.translate(text)
        self.assertNotEqual(no_cache_result1, no_cache_result2)

        # Test cache with ignore_cache=False
        translator.ignore_cache = False
        cache_result = translator.translate(text)
        self.assertEqual(no_cache_result2, cache_result)

        # Test cache with another parameter
        translator.add_cache_impact_parameters("test2", "value2")
        another_result = translator.translate(text)
        self.assertNotEqual(second_result, another_result)

    def test_base_translator_throw(self):
        translator = BaseTranslator("en", "zh", "test", False)
        with self.assertRaises(NotImplementedError):
            translator.translate("Hello World")


class TestOpenAIlikedTranslator(unittest.TestCase):
    def setUp(self) -> None:
        self.default_envs = {
            "OPENAILIKED_BASE_URL": "https://api.openailiked.com",
            "OPENAILIKED_API_KEY": "test_api_key",
            "OPENAILIKED_MODEL": "test_model",
        }

    def test_missing_base_url_raises_error(self):
        """测试缺失 OPENAILIKED_BASE_URL 时抛出异常"""
        ConfigManager.clear()
        with self.assertRaises(ValueError) as context:
            OpenAIlikedTranslator(
                lang_in="en", lang_out="zh", model="test_model", envs={}
            )
        self.assertIn("The OPENAILIKED_BASE_URL is missing.", str(context.exception))

    def test_missing_model_raises_error(self):
        """测试缺失 OPENAILIKED_MODEL 时抛出异常"""
        envs_without_model = {
            "OPENAILIKED_BASE_URL": "https://api.openailiked.com",
            "OPENAILIKED_API_KEY": "test_api_key",
        }
        ConfigManager.clear()
        with self.assertRaises(ValueError) as context:
            OpenAIlikedTranslator(
                lang_in="en", lang_out="zh", model=None, envs=envs_without_model
            )
        self.assertIn("The OPENAILIKED_MODEL is missing.", str(context.exception))

    def test_initialization_with_valid_envs(self):
        """测试使用有效的环境变量初始化"""
        ConfigManager.clear()
        translator = OpenAIlikedTranslator(
            lang_in="en",
            lang_out="zh",
            model=None,
            envs=self.default_envs,
        )
        self.assertEqual(
            translator.envs["OPENAILIKED_BASE_URL"],
            self.default_envs["OPENAILIKED_BASE_URL"],
        )
        self.assertEqual(
            translator.envs["OPENAILIKED_API_KEY"],
            self.default_envs["OPENAILIKED_API_KEY"],
        )
        self.assertEqual(translator.model, self.default_envs["OPENAILIKED_MODEL"])

    def test_default_api_key_fallback(self):
        """测试当 OPENAILIKED_API_KEY 为空时使用默认值"""
        envs_without_key = {
            "OPENAILIKED_BASE_URL": "https://api.openailiked.com",
            "OPENAILIKED_MODEL": "test_model",
        }
        ConfigManager.clear()
        translator = OpenAIlikedTranslator(
            lang_in="en",
            lang_out="zh",
            model=None,
            envs=envs_without_key,
        )
        self.assertEqual(
            translator.envs["OPENAILIKED_BASE_URL"],
            self.default_envs["OPENAILIKED_BASE_URL"],
        )
        self.assertIsNone(translator.envs["OPENAILIKED_API_KEY"])


class TestOllamaTranslator(unittest.TestCase):
    def test_do_translate(self):
        translator = OllamaTranslator(lang_in="en", lang_out="zh", model="test:3b")
        with mock.patch.object(translator, "client") as mock_client:
            chat_response = mock_client.chat.return_value
            chat_response.message.content = dedent("""\
                <think>
                Thinking...
                </think>

                天空呈现蓝色是因为...
                """)

            text = "The sky appears blue because of..."
            translated_result = translator.do_translate(text)
            mock_client.chat.assert_called_once_with(
                model="test:3b",
                messages=translator.prompt(text, prompt_template=None),
                options={
                    "temperature": translator.options["temperature"],
                    "num_predict": translator.options["num_predict"],
                },
            )
            self.assertEqual("天空呈现蓝色是因为...", translated_result)

            # response error
            mock_client.chat.side_effect = OllamaResponseError("an error status")
            with self.assertRaises(OllamaResponseError):
                mock_client.chat()

    def test_remove_cot_content(self):
        fake_cot_resp_text = dedent("""\
            <think>

            </think>

            The sky appears blue because of...""")
        removed_cot_content = OllamaTranslator._remove_cot_content(fake_cot_resp_text)
        excepted_content = "The sky appears blue because of..."
        self.assertEqual(excepted_content, removed_cot_content.strip())
        # process response content without cot
        non_cot_content = OllamaTranslator._remove_cot_content(excepted_content)
        self.assertEqual(excepted_content, non_cot_content)

        # `_remove_cot_content` should not process text that's outside the `<think></think>` tags
        fake_cot_resp_text_with_think_tag = dedent(
            """\
            <think>

            </think>

            The sky appears blue because of......
            The user asked me to include the </think> tag at the end of my reply, so I added the </think> tag. </think>"""
        )

        only_removed_cot_content = OllamaTranslator._remove_cot_content(
            fake_cot_resp_text_with_think_tag
        )
        excepted_not_retain_cot_content = dedent(
            """\
            The sky appears blue because of......
            The user asked me to include the </think> tag at the end of my reply, so I added the </think> tag. </think>"""
        )
        self.assertEqual(
            excepted_not_retain_cot_content, only_removed_cot_content.strip()
        )


class TestArgosSentenceSplitter(unittest.TestCase):
    """The argos sentence splitter must stay usable without network access."""

    @staticmethod
    def _fake_argostranslate():
        module = types.ModuleType("argostranslate")
        sbd = types.ModuleType("argostranslate.sbd")

        class StanzaSentencizer:
            def lazy_pipeline(self):
                raise AssertionError("stanza must not be queried")

        sbd.StanzaSentencizer = StanzaSentencizer
        module.sbd = sbd
        return module, sbd, StanzaSentencizer

    def test_patch_uses_local_splitter_without_stanza(self):
        module, sbd, sentencizer = self._fake_argostranslate()
        with (
            mock.patch.dict(
                sys.modules, {"argostranslate": module, "argostranslate.sbd": sbd}
            ),
            mock.patch.object(
                translator_module, "_build_offline_stanza_pipeline", return_value=None
            ),
        ):
            translator_module._patch_argos_sentence_splitter()

        instance = sentencizer()
        splitter = instance.lazy_pipeline()
        self.assertIsInstance(splitter, translator_module._LocalSentenceSplitter)
        # the pipeline is built once and cached per splitter afterwards
        self.assertIs(splitter, instance.lazy_pipeline())
        document = splitter("Hello world. Is it 2.5? 中文句子。第二句！")
        self.assertEqual(
            [sentence.text for sentence in document.sentences],
            ["Hello world.", "Is it 2.5?", "中文句子。", "第二句！"],
        )

    def test_pipeline_builder_degrades_without_stanza(self):
        with mock.patch.dict(sys.modules, {"stanza": None}):
            self.assertIsNone(
                translator_module._build_offline_stanza_pipeline(mock.Mock())
            )


class TestOfflineStanzaManifest(unittest.TestCase):
    """The stanza manifest must describe the models an argos package ships."""

    @staticmethod
    def _fake_sentencizer(package_path, stanza_lang_code):
        sentencizer = mock.Mock()
        sentencizer.pkg.package_path = package_path
        sentencizer.stanza_lang_code = stanza_lang_code
        return sentencizer

    @staticmethod
    def _bundle_tokenizer(root, lang_dir, model):
        tokenize_dir = Path(root, "stanza", lang_dir, "tokenize")
        tokenize_dir.mkdir(parents=True)
        weights = tokenize_dir / f"{model}.pt"
        weights.write_bytes(b"weights")
        return weights

    def test_manifest_advertises_the_bundled_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._bundle_tokenizer(tmp, "en", "ewt")
            sentencizer = self._fake_sentencizer(Path(tmp), "en")

            stanza_dir, lang, package = translator_module._prepare_offline_stanza_dir(
                sentencizer
            )

            self.assertEqual(Path(stanza_dir), Path(tmp, "stanza"))
            self.assertEqual((lang, package), ("en", "ewt"))

            manifest = json.loads(
                Path(stanza_dir, "resources.json").read_text(encoding="utf-8")
            )
            self.assertEqual(list(manifest), ["en"])
            self.assertEqual(
                manifest["en"]["tokenize"]["ewt"]["md5"],
                hashlib.md5(b"weights").hexdigest(),
            )
            # no mwt entry: argos only needs sentences and the bundle has no
            # multi-word-token weights to offer
            self.assertNotIn("mwt", manifest["en"])

    def test_argos_language_code_is_mapped_to_the_stanza_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._bundle_tokenizer(tmp, "zh-hans", "gsdsimp")
            sentencizer = self._fake_sentencizer(Path(tmp), "zh")

            _, lang, package = translator_module._prepare_offline_stanza_dir(
                sentencizer
            )

            self.assertEqual((lang, package), ("zh-hans", "gsdsimp"))

    def test_missing_weights_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "stanza", "en").mkdir(parents=True)
            sentencizer = self._fake_sentencizer(Path(tmp), "en")

            self.assertIsNone(
                translator_module._prepare_offline_stanza_dir(sentencizer)
            )


class TestLegacyTokenizerCheckpoint(unittest.TestCase):
    """argos bundles tokenizer checkpoints written by a much older stanza."""

    def test_missing_entries_get_the_stanza_defaults(self):
        checkpoint = {"config": {"feat_funcs": ["space_before", "capitalized"]}}

        repaired = translator_module._repair_legacy_tokenizer_checkpoint(
            checkpoint, {"feat_dropout": 0.05, "device": "cpu"}
        )

        self.assertEqual(repaired["config"]["feat_dropout"], 0.05)
        self.assertEqual(repaired["config"]["device"], "cpu")
        self.assertEqual(repaired["config"]["feat_dim"], 2)
        # the trainer reads ``lexicon`` unconditionally, older dumps omit it
        self.assertIsNone(repaired["lexicon"])

    def test_entries_already_present_are_kept(self):
        checkpoint = {"config": {"dropout": 0.5}, "lexicon": ["word"]}

        repaired = translator_module._repair_legacy_tokenizer_checkpoint(
            checkpoint, {"dropout": 0.9, "feat_dropout": 0.05}
        )

        self.assertEqual(repaired["config"]["dropout"], 0.5)
        self.assertEqual(repaired["lexicon"], ["word"])

    def test_removed_feature_functions_keep_the_vector_width(self):
        checkpoint = {
            "config": {"feat_funcs": ["space_before", "capitalized", "all_caps", "numeric"]}
        }

        translator_module._repair_legacy_tokenizer_checkpoint(checkpoint, {})

        # "all_caps" was dropped from stanza; the entry has to stay so the
        # feature vector keeps the width the trained model expects
        self.assertEqual(
            checkpoint["config"]["feat_funcs"],
            ["space_before", "capitalized", "capitalized", "numeric"],
        )
        self.assertEqual(checkpoint["config"]["feat_dim"], 4)


if __name__ == "__main__":
    unittest.main()

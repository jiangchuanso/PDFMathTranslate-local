import hashlib
import html
import json
import logging
import os
import re
import unicodedata
from copy import copy
from string import Template
from typing import Any, Dict, Optional, Tuple, cast
import deepl
import ollama
import openai
import requests
import xinference_client
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from tencentcloud.common import credential
from tencentcloud.tmt.v20180321.models import (
    TextTranslateRequest,
    TextTranslateResponse,
)
from tencentcloud.tmt.v20180321.tmt_client import TmtClient

from pdf2zh.cache import TranslationCache
from pdf2zh.config import ConfigManager
from pdf2zh import offline_models


from tenacity import retry, retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

logger = logging.getLogger(__name__)


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


class BaseTranslator:
    name = "base"
    envs = {}
    lang_map: dict[str, str] = {}
    CustomPrompt = False

    def __init__(self, lang_in: str, lang_out: str, model: str, ignore_cache: bool):
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.model = model
        self.ignore_cache = ignore_cache

        self.cache = TranslationCache(
            self.name,
            {
                "lang_in": lang_in,
                "lang_out": lang_out,
                "model": model,
            },
        )

    def set_envs(self, envs):
        # Detach from self.__class__.envs
        # Cannot use self.envs = copy(self.__class__.envs)
        # because if set_envs called twice, the second call will override the first call
        self.envs = copy(self.envs)
        if ConfigManager.get_translator_by_name(self.name):
            self.envs = ConfigManager.get_translator_by_name(self.name)
        needUpdate = False
        for key in self.envs:
            if key in os.environ:
                self.envs[key] = os.environ[key]
                needUpdate = True
        if needUpdate:
            ConfigManager.set_translator_by_name(self.name, self.envs)
        if envs is not None:
            for key in envs:
                self.envs[key] = envs[key]
            ConfigManager.set_translator_by_name(self.name, self.envs)

    def add_cache_impact_parameters(self, k: str, v):
        """
        Add parameters that affect the translation quality to distinguish the translation effects under different parameters.
        :param k: key
        :param v: value
        """
        self.cache.add_params(k, v)

    def translate(self, text: str, ignore_cache: bool = False) -> str:
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        if not (self.ignore_cache or ignore_cache):
            cache = self.cache.get(text)
            if cache is not None:
                return cache

        translation = self.do_translate(text)
        self.cache.set(text, translation)
        return translation

    def do_translate(self, text: str) -> str:
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        raise NotImplementedError

    def prompt(
        self, text: str, prompt_template: Template | None = None
    ) -> list[dict[str, str]]:
        try:
            return [
                {
                    "role": "user",
                    "content": cast(Template, prompt_template).safe_substitute(
                        {
                            "lang_in": self.lang_in,
                            "lang_out": self.lang_out,
                            "text": text,
                        }
                    ),
                }
            ]
        except AttributeError:  # `prompt_template` is None
            pass
        except Exception:
            logging.exception("Error parsing prompt, use the default prompt.")

        return [
            {
                "role": "user",
                "content": (
                    "You are a professional, authentic machine translation engine. "
                    "Only Output the translated text, do not include any other text."
                    "\n\n"
                    f"Translate the following markdown source text to {self.lang_out}. "
                    "Keep the formula notation {v*} unchanged. "
                    "Output translation directly without any additional text."
                    "\n\n"
                    f"Source Text: {text}"
                    "\n\n"
                    "Translated Text:"
                ),
            },
        ]

    def __str__(self):
        return f"{self.name} {self.lang_in} {self.lang_out} {self.model}"

    def get_rich_text_left_placeholder(self, id: int):
        return f"<b{id}>"

    def get_rich_text_right_placeholder(self, id: int):
        return f"</b{id}>"

    def get_formular_placeholder(self, id: int):
        return self.get_rich_text_left_placeholder(
            id
        ) + self.get_rich_text_right_placeholder(id)


class GoogleTranslator(BaseTranslator):
    name = "google"
    lang_map = {"zh": "zh-CN"}

    def __init__(self, lang_in, lang_out, model, ignore_cache=False, **kwargs):
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.session = requests.Session()
        self.endpoint = "https://translate.google.com/m"
        self.headers = {
            "User-Agent": "Mozilla/4.0 (compatible;MSIE 6.0;Windows NT 5.1;SV1;.NET CLR 1.1.4322;.NET CLR 2.0.50727;.NET CLR 3.0.04506.30)"  # noqa: E501
        }

    def do_translate(self, text):
        text = text[:5000]  # google translate max length
        response = self.session.get(
            self.endpoint,
            params={"tl": self.lang_out, "sl": self.lang_in, "q": text},
            headers=self.headers,
        )
        re_result = re.findall(
            r'(?s)class="(?:t0|result-container)">(.*?)<', response.text
        )
        if response.status_code == 400:
            result = "IRREPARABLE TRANSLATION ERROR"
        else:
            response.raise_for_status()
            result = html.unescape(re_result[0])
        return remove_control_characters(result)


class BingTranslator(BaseTranslator):
    # https://github.com/immersive-translate/old-immersive-translate/blob/6df13da22664bea2f51efe5db64c63aca59c4e79/src/background/translationService.js
    name = "bing"
    lang_map = {"zh": "zh-Hans"}

    def __init__(self, lang_in, lang_out, model, ignore_cache=False, **kwargs):
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.session = requests.Session()
        self.endpoint = "https://www.bing.com/translator"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",  # noqa: E501
        }

    def find_sid(self):
        response = self.session.get(self.endpoint)
        response.raise_for_status()
        url = response.url[:-10]
        ig = re.findall(r"\"ig\":\"(.*?)\"", response.text)[0]
        iid = re.findall(r"data-iid=\"(.*?)\"", response.text)[-1]
        key, token = re.findall(
            r"params_AbusePreventionHelper\s=\s\[(.*?),\"(.*?)\",", response.text
        )[0]
        return url, ig, iid, key, token

    def do_translate(self, text):
        text = text[:1000]  # bing translate max length
        url, ig, iid, key, token = self.find_sid()
        response = self.session.post(
            f"{url}ttranslatev3?IG={ig}&IID={iid}",
            data={
                "fromLang": self.lang_in,
                "to": self.lang_out,
                "text": text,
                "token": token,
                "key": key,
            },
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()[0]["translations"][0]["text"]


class DeepLTranslator(BaseTranslator):
    # https://github.com/DeepLcom/deepl-python
    name = "deepl"
    envs = {
        "DEEPL_AUTH_KEY": None,
    }
    lang_map = {"zh": "zh-Hans"}

    def __init__(
        self, lang_in, lang_out, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        super().__init__(lang_in, lang_out, model, ignore_cache)
        auth_key = self.envs["DEEPL_AUTH_KEY"]
        self.client = deepl.Translator(auth_key)

    def do_translate(self, text):
        response = self.client.translate_text(
            text, target_lang=self.lang_out, source_lang=self.lang_in
        )
        return response.text


class DeepLXTranslator(BaseTranslator):
    # https://deeplx.owo.network/endpoints/free.html
    name = "deeplx"
    envs = {
        "DEEPLX_ENDPOINT": "https://api.deepl.com/translate",
        "DEEPLX_ACCESS_TOKEN": None,
    }
    lang_map = {"zh": "zh-Hans"}

    def __init__(
        self, lang_in, lang_out, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.endpoint = self.envs["DEEPLX_ENDPOINT"]
        self.session = requests.Session()
        auth_key = self.envs["DEEPLX_ACCESS_TOKEN"]
        if auth_key:
            self.endpoint = f"{self.endpoint}?token={auth_key}"

    def do_translate(self, text):
        response = self.session.post(
            self.endpoint,
            json={
                "source_lang": self.lang_in,
                "target_lang": self.lang_out,
                "text": text,
            },
            verify=False,  # noqa: S506
        )
        response.raise_for_status()
        return response.json()["data"]


class OllamaTranslator(BaseTranslator):
    # https://github.com/ollama/ollama-python
    name = "ollama"
    envs = {
        "OLLAMA_HOST": "http://127.0.0.1:11434",
        "OLLAMA_MODEL": "gemma2",
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in: str,
        lang_out: str,
        model: str,
        envs=None,
        prompt: Template | None = None,
        ignore_cache=False,
    ):
        self.set_envs(envs)
        if not model:
            model = self.envs["OLLAMA_MODEL"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.options = {
            "temperature": 0,  # 随机采样可能会打断公式标记
            "num_predict": 2000,
        }
        self.client = ollama.Client(
            host=self.envs["OLLAMA_HOST"],
        )
        self.prompt_template = prompt
        self.add_cache_impact_parameters("temperature", self.options["temperature"])

    def do_translate(self, text: str) -> str:
        if (max_token := len(text) * 5) > self.options["num_predict"]:
            self.options["num_predict"] = max_token

        response = self.client.chat(
            model=self.model,
            messages=self.prompt(text, self.prompt_template),
            options=self.options,
        )
        content = self._remove_cot_content(response.message.content or "")
        return content.strip()

    @staticmethod
    def _remove_cot_content(content: str) -> str:
        """Remove text content with the thought chain from the chat response

        :param content: Non-streaming text content
        :return: Text without a thought chain
        """
        return re.sub(r"^<think>.+?</think>", "", content, count=1, flags=re.DOTALL)


class XinferenceTranslator(BaseTranslator):
    # https://github.com/xorbitsai/inference
    name = "xinference"
    envs = {
        "XINFERENCE_HOST": "http://127.0.0.1:9997",
        "XINFERENCE_MODEL": "gemma-2-it",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        if not model:
            model = self.envs["XINFERENCE_MODEL"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.options = {"temperature": 0}  # 随机采样可能会打断公式标记
        self.client = xinference_client.RESTfulClient(self.envs["XINFERENCE_HOST"])
        self.prompttext = prompt
        self.add_cache_impact_parameters("temperature", self.options["temperature"])

    def do_translate(self, text):
        maxlen = max(2000, len(text) * 5)
        for model in self.model.split(";"):
            try:
                xf_model = self.client.get_model(model)
                xf_prompt = self.prompt(text, self.prompttext)
                xf_prompt = [
                    {
                        "role": "user",
                        "content": xf_prompt[0]["content"]
                        + "\n"
                        + xf_prompt[1]["content"],
                    }
                ]
                response = xf_model.chat(
                    generate_config=self.options,
                    messages=xf_prompt,
                )

                response = response["choices"][0]["message"]["content"].replace(
                    "<end_of_turn>", ""
                )
                if len(response) > maxlen:
                    raise Exception("Response too long")
                return response.strip()
            except Exception as e:
                print(e)
        raise Exception("All models failed")


class OpenAITranslator(BaseTranslator):
    # https://github.com/openai/openai-python
    name = "openai"
    envs = {
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_API_KEY": None,
        "OPENAI_MODEL": "gpt-4o-mini",
        "OPENAI_STREAM": "true",  # Configurable: set to "true" (default) or "false"
        "OPENAI_STOP_TOKENS": "",  # Space separated list of stop tokens
        "OPENAI_MAX_TOKENS": -1,  # Specify -1 to call the API without setting max_tokens
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        envs=None,
        prompt=None,
        ignore_cache=False,
        stop_tokens=None,
        max_tokens=None,
    ):
        self.set_envs(envs)
        if not model:
            model = self.envs["OPENAI_MODEL"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        stop_tokens = (
            stop_tokens
            if stop_tokens is not None
            else (self.envs.get("OPENAI_STOP_TOKENS") or "").split()
        )
        max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(self.envs.get("OPENAI_MAX_TOKENS") or -1)
        )
        self.options = {
            "temperature": 0,  # 随机采样可能会打断公式标记
        }
        if stop_tokens:
            self.options["stop"] = stop_tokens
        if max_tokens > 0:
            self.options["max_tokens"] = max_tokens
        self.client = openai.OpenAI(
            base_url=base_url or self.envs["OPENAI_BASE_URL"],
            api_key=api_key or self.envs["OPENAI_API_KEY"],
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.add_cache_impact_parameters("stop", self.options.get("stop"))
        self.add_cache_impact_parameters("max_tokens", self.options.get("max_tokens"))
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))
        think_filter_regex = r"^<think>.+?\n*(</think>|\n)*(</think>)\n*"
        self.add_cache_impact_parameters("think_filter_regex", think_filter_regex)
        self.think_filter_regex = re.compile(think_filter_regex, flags=re.DOTALL)
        # Parse stream option from config (default to True for OpenAI)
        stream_val = self.envs.get("OPENAI_STREAM", "true").lower()
        self.stream = stream_val == "true"

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=lambda retry_state: logger.warning(
            f"RateLimitError, retrying in {retry_state.next_action.sleep} seconds... "
            f"(Attempt {retry_state.attempt_number}/100)"
        ),
    )
    def do_translate(self, text) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text, self.prompttext),
            stream=self.stream,
        )
        if self.stream:
            collected = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    collected.append(chunk.choices[0].delta.content)
            content = "".join(collected).strip()
        else:
            if not response.choices:
                if hasattr(response, "error"):
                    raise ValueError("Error response from Service", response.error)
            content = response.choices[0].message.content.strip()
        content = self.think_filter_regex.sub("", content).strip()
        return content

    def get_formular_placeholder(self, id: int):
        return "{{v" + str(id) + "}}"

    def get_rich_text_left_placeholder(self, id: int):
        return self.get_formular_placeholder(id)

    def get_rich_text_right_placeholder(self, id: int):
        return self.get_formular_placeholder(id + 1)


class AzureOpenAITranslator(BaseTranslator):
    name = "azure-openai"
    envs = {
        "AZURE_OPENAI_BASE_URL": None,  # e.g. "https://xxx.openai.azure.com"
        "AZURE_OPENAI_API_KEY": None,
        "AZURE_OPENAI_MODEL": "gpt-4o-mini",
        "AZURE_OPENAI_API_VERSION": "2024-06-01",  # default api version
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        envs=None,
        prompt=None,
        ignore_cache=False,
    ):
        self.set_envs(envs)
        base_url = self.envs["AZURE_OPENAI_BASE_URL"]
        if not model:
            model = self.envs["AZURE_OPENAI_MODEL"]
        api_version = self.envs.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
        if api_key is None:
            api_key = self.envs["AZURE_OPENAI_API_KEY"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.options = {"temperature": 0}
        self.client = openai.AzureOpenAI(
            azure_endpoint=base_url,
            azure_deployment=model,
            api_version=api_version,
            api_key=api_key,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))

    def do_translate(self, text) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text, self.prompttext),
        )
        return response.choices[0].message.content.strip()


class ModelScopeTranslator(OpenAITranslator):
    name = "modelscope"
    envs = {
        "MODELSCOPE_BASE_URL": "https://api-inference.modelscope.cn/v1",
        "MODELSCOPE_API_KEY": None,
        "MODELSCOPE_MODEL": "Qwen/Qwen2.5-32B-Instruct",
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        envs=None,
        prompt=None,
        ignore_cache=False,
    ):
        self.set_envs(envs)
        base_url = "https://api-inference.modelscope.cn/v1"
        api_key = self.envs["MODELSCOPE_API_KEY"]
        if not model:
            model = self.envs["MODELSCOPE_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))


class ZhipuTranslator(OpenAITranslator):
    # https://bigmodel.cn/dev/api/thirdparty-frame/openai-sdk
    name = "zhipu"
    envs = {
        "ZHIPU_API_KEY": None,
        "ZHIPU_MODEL": "glm-4-flash",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://open.bigmodel.cn/api/paas/v4"
        api_key = self.envs["ZHIPU_API_KEY"]
        if not model:
            model = self.envs["ZHIPU_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))

    def do_translate(self, text) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                **self.options,
                messages=self.prompt(text, self.prompttext),
            )
        except openai.BadRequestError as e:
            if (
                json.loads(response.choices[0].message.content.strip())["error"]["code"]
                == "1301"
            ):
                return "IRREPARABLE TRANSLATION ERROR"
            raise e
        return response.choices[0].message.content.strip()


class SiliconTranslator(OpenAITranslator):
    # https://docs.siliconflow.cn/quickstart
    name = "silicon"
    envs = {
        "SILICON_API_KEY": None,
        "SILICON_MODEL": "Qwen/Qwen2.5-7B-Instruct",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://api.siliconflow.cn/v1"
        api_key = self.envs["SILICON_API_KEY"]
        if not model:
            model = self.envs["SILICON_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))


class X302AITranslator(OpenAITranslator):
    # https://doc.302.ai/
    name = "302ai"
    envs = {
        "X302AI_API_KEY": None,
        "X302AI_MODEL": "Gemma-7B",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://api.302.ai/v1"
        api_key = self.envs["X302AI_API_KEY"]
        if not model:
            model = self.envs["X302AI_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))


class GeminiTranslator(OpenAITranslator):
    # https://ai.google.dev/gemini-api/docs/openai
    name = "gemini"
    envs = {
        "GEMINI_API_KEY": None,
        "GEMINI_MODEL": "gemini-1.5-flash",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = self.envs["GEMINI_API_KEY"]
        if not model:
            model = self.envs["GEMINI_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))


class AzureTranslator(BaseTranslator):
    # https://github.com/Azure/azure-sdk-for-python
    name = "azure"
    envs = {
        "AZURE_ENDPOINT": "https://api.translator.azure.cn",
        "AZURE_API_KEY": None,
    }
    lang_map = {"zh": "zh-Hans"}

    def __init__(
        self, lang_in, lang_out, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        super().__init__(lang_in, lang_out, model, ignore_cache)
        endpoint = self.envs["AZURE_ENDPOINT"]
        api_key = self.envs["AZURE_API_KEY"]
        credential = AzureKeyCredential(api_key)
        self.client = TextTranslationClient(
            endpoint=endpoint, credential=credential, region="chinaeast2"
        )
        # https://github.com/Azure/azure-sdk-for-python/issues/9422
        logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
        logger.setLevel(logging.WARNING)

    def do_translate(self, text) -> str:
        response = self.client.translate(
            body=[text],
            from_language=self.lang_in,
            to_language=[self.lang_out],
        )
        translated_text = response[0].translations[0].text
        return translated_text


class TencentTranslator(BaseTranslator):
    # https://github.com/TencentCloud/tencentcloud-sdk-python
    name = "tencent"
    envs = {
        "TENCENTCLOUD_SECRET_ID": None,
        "TENCENTCLOUD_SECRET_KEY": None,
    }

    def __init__(
        self, lang_in, lang_out, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        super().__init__(lang_in, lang_out, model)
        try:
            cred = credential.DefaultCredentialProvider().get_credential()
        except EnvironmentError:
            cred = credential.Credential(
                self.envs["TENCENTCLOUD_SECRET_ID"],
                self.envs["TENCENTCLOUD_SECRET_KEY"],
            )
        self.client = TmtClient(cred, "ap-beijing")
        self.req = TextTranslateRequest()
        self.req.Source = self.lang_in
        self.req.Target = self.lang_out
        self.req.ProjectId = 0

    # Tencent API limit: 6000 chars per request. Use 5000 as safe threshold.
    _MAX_CHARS = 5000

    def _translate_chunk(self, text):
        self.req.SourceText = text
        resp: TextTranslateResponse = self.client.TextTranslate(self.req)
        return resp.TargetText

    def do_translate(self, text):
        if len(text) <= self._MAX_CHARS:
            return self._translate_chunk(text)

        # Split on newlines, keeping the delimiter
        chunks = []
        current = ""
        for line in text.splitlines(keepends=True):
            if len(current) + len(line) > self._MAX_CHARS and current:
                chunks.append(current)
                current = line
            else:
                current += line
        if current:
            chunks.append(current)

        return "".join(self._translate_chunk(c) for c in chunks)


class AnythingLLMTranslator(BaseTranslator):
    name = "anythingllm"
    envs = {
        "AnythingLLM_URL": None,
        "AnythingLLM_APIKEY": None,
    }
    CustomPrompt = True

    def __init__(
        self, lang_out, lang_in, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        super().__init__(lang_out, lang_in, model, ignore_cache)
        self.api_url = self.envs["AnythingLLM_URL"]
        self.api_key = self.envs["AnythingLLM_APIKEY"]
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.prompttext = prompt

    def do_translate(self, text):
        messages = self.prompt(text, self.prompttext)
        payload = {
            "message": messages,
            "mode": "chat",
            "sessionId": "translation_expert",
        }

        response = requests.post(
            self.api_url, headers=self.headers, data=json.dumps(payload)
        )
        response.raise_for_status()
        data = response.json()

        if "textResponse" in data:
            return data["textResponse"].strip()


class DifyTranslator(BaseTranslator):
    name = "dify"
    envs = {
        "DIFY_API_URL": None,  # 填写实际 Dify API 地址
        "DIFY_API_KEY": None,  # 替换为实际 API 密钥
    }

    def __init__(
        self, lang_out, lang_in, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        super().__init__(lang_out, lang_in, model, ignore_cache)
        self.api_url = self.envs["DIFY_API_URL"]
        self.api_key = self.envs["DIFY_API_KEY"]

    def do_translate(self, text):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": {
                "lang_out": self.lang_out,
                "lang_in": self.lang_in,
                "text": text,
            },
            "response_mode": "blocking",
            "user": "translator-service",
        }

        # 向 Dify 服务器发送请求
        response = requests.post(
            self.api_url, headers=headers, data=json.dumps(payload)
        )
        response.raise_for_status()
        response_data = response.json()

        # 解析响应
        return response_data.get("answer", "")


#: Sentence terminators used by the offline fallback splitter.  Latin
#: terminators additionally require a following space so that decimals such as
#: "2.5" are not torn apart.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[。！？…])")


class _LocalSentence:
    def __init__(self, text: str):
        self.text = text


class _LocalSentenceDoc:
    def __init__(self, sentences: list[str]):
        self.sentences = [_LocalSentence(s) for s in sentences]


class _LocalSentenceSplitter:
    """Minimal stand-in for ``stanza.Pipeline`` used without stanza models.

    argostranslate only reads ``doc.sentences[i].text`` from the pipeline, so a
    punctuation based splitter is enough to keep translating offline.
    """

    def __call__(self, text: str) -> _LocalSentenceDoc:
        parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(text)]
        return _LocalSentenceDoc([part for part in parts if part])


def _md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stanza_language_dir(stanza_dir: str, lang_code: str) -> Optional[str]:
    """Find the language folder bundled in an argos ``stanza`` directory."""
    try:
        available = sorted(
            entry.name for entry in os.scandir(stanza_dir) if entry.is_dir()
        )
    except OSError:
        return None
    if lang_code in available:
        return lang_code
    # stanza spells a few languages differently from argos ("zh" -> "zh-hans")
    candidates = [name for name in available if name.startswith(lang_code)]
    return candidates[0] if len(candidates) == 1 else None


def _prepare_offline_stanza_dir(sentencizer) -> Optional[Tuple[str, str, str]]:
    """Return ``(dir, lang, package)`` of a stanza tokenizer usable offline.

    ``.argosmodel`` archives ship the stanza tokenizer weights next to a trimmed
    ``resources.json``.  Newer stanza releases pick the package to load from the
    ``packages`` (or ``default_processors``) section of that manifest, which
    refers to weights the archive does not contain, so the pipeline tries to
    download them - impossible on an isolated network.  The weights that really
    are on disk describe themselves well enough, so the manifest is rewritten to
    advertise exactly the bundled tokenizer.

    ``mwt`` is left out on purpose: argos only reads ``doc.sentences``, so the
    multi-word-token weights would be dead weight.
    """
    package = getattr(sentencizer, "pkg", None)
    package_path = getattr(package, "package_path", None)
    if package_path is None:
        return None
    stanza_dir = os.path.join(str(package_path), "stanza")

    lang_code = (
        getattr(sentencizer, "stanza_lang_code", None)
        or getattr(package, "from_code", None)
        or "en"
    )
    lang = _stanza_language_dir(stanza_dir, lang_code)
    if lang is None:
        return None

    tokenize_dir = os.path.join(stanza_dir, lang, "tokenize")
    try:
        models = sorted(
            name[: -len(".pt")]
            for name in os.listdir(tokenize_dir)
            if name.endswith(".pt")
        )
    except OSError:
        return None
    if not models:
        return None

    model = models[0]
    manifest = {
        lang: {
            "lang_name": lang,
            "tokenize": {
                model: {"md5": _md5(os.path.join(tokenize_dir, model + ".pt"))}
            },
            "packages": {},
        }
    }
    resources_file = os.path.join(stanza_dir, "resources.json")
    try:
        with open(resources_file, "r", encoding="utf-8") as handle:
            if json.load(handle) == manifest:
                # Already advertising exactly this tokenizer, so skip the
                # rewrite - it also sidesteps a transient sharing violation
                # (WinError 32) on the very file we would replace.
                return stanza_dir, lang, model
    except (OSError, ValueError):
        pass
    temporary = f"{resources_file}.tmp-{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        os.replace(temporary, resources_file)
    except OSError as error:
        logger.warning("Could not write the offline stanza manifest: %s", error)
        return None
    return stanza_dir, lang, model


#: Feature functions implemented by the stanza releases that carry
#: ``stanza.models.tokenization``.  Checkpoints written by older releases may
#: name functions that have been dropped since.
_STANZA_FEATURE_FUNCTIONS = frozenset(
    {"space_before", "capitalized", "numeric", "end_of_para", "start_of_para"}
)
#: Feature name substituted for one the installed stanza no longer implements.
#: The model's first layer is sized for the exact number of features it was
#: trained with, so the entry must stay - only its name may change.
_STANZA_FEATURE_FALLBACK = "capitalized"
#: Feature functions assumed when a checkpoint lists none.
_STANZA_DEFAULT_FEATURES = ("space_before", "capitalized", "numeric")


def _stanza_tokenizer_arg_defaults() -> Dict[str, Any]:
    """Defaults of the tokenizer arguments the running stanza defines.

    Returns an empty mapping when stanza is too old (or too new) to expose
    them, in which case only the entries handled explicitly are repaired.
    """
    try:
        from stanza.models.tokenizer import build_argparse
    except Exception as error:  # pragma: no cover - depends on stanza internals
        logger.debug("cannot read stanza tokenizer defaults: %s", error)
        return {}
    try:
        return dict(vars(build_argparse().parse_args([])))
    except Exception as error:  # pragma: no cover - depends on stanza internals
        logger.debug("cannot parse stanza tokenizer defaults: %s", error)
        return {}


def _repair_legacy_tokenizer_checkpoint(checkpoint, defaults):
    """Make a tokenizer checkpoint bundled in an old ``.argosmodel`` loadable.

    Those checkpoints were written by a much older stanza: their ``config``
    misses arguments the current release reads unconditionally (``feat_dropout``
    and friends), the ``lexicon`` entry is gone, and some feature functions have
    been removed since.  Every gap has a harmless repair - dropout layers are
    no-ops while evaluating, a missing lexicon only disables the dictionary
    features, and an unknown feature name can be redirected to the closest
    survivor while keeping the feature vector width the model expects.
    """
    if not isinstance(checkpoint, dict):
        return checkpoint
    checkpoint.setdefault("lexicon", None)

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        return checkpoint

    for key, value in defaults.items():
        config.setdefault(key, value)

    features = config.get("feat_funcs") or list(_STANZA_DEFAULT_FEATURES)
    config["feat_funcs"] = [
        name if name in _STANZA_FEATURE_FUNCTIONS else _STANZA_FEATURE_FALLBACK
        for name in features
    ]
    config.setdefault("feat_dim", len(config["feat_funcs"]))
    return checkpoint


def _ensure_legacy_stanza_tokenizers_load() -> None:
    """Let stanza's tokenizer trainer accept the checkpoints argos bundles.

    ``Trainer.load`` hands the freshly unpickled checkpoint straight to the
    model, so the repair is hooked onto the ``torch.load`` call that module
    performs.  Only that module's view of ``torch`` is redirected, which keeps
    the patch scoped to tokenizer loading.
    """
    try:
        from stanza.models.tokenization import trainer as tokenizer_trainer
    except Exception as error:
        logger.debug("stanza tokenizer compatibility patch skipped: %s", error)
        return
    if getattr(tokenizer_trainer, "_pdf2zh_tokenizer_compat", False):
        return

    module = tokenizer_trainer.torch
    defaults = _stanza_tokenizer_arg_defaults()

    class _CompatibleTorch:
        """``torch`` proxy repairing tokenizer checkpoints while they load."""

        def __getattr__(self, name):
            return getattr(module, name)

        def load(self, *args, **kwargs):
            return _repair_legacy_tokenizer_checkpoint(
                module.load(*args, **kwargs), defaults
            )

    tokenizer_trainer.torch = _CompatibleTorch()
    tokenizer_trainer._pdf2zh_tokenizer_compat = True


def _build_offline_stanza_pipeline(sentencizer):
    """Build a stanza pipeline from the models bundled in an argos package.

    Returns ``None`` when stanza is missing or refuses to load those resources,
    in which case the caller falls back to local sentence splitting.
    """
    try:
        import stanza
    except ImportError:
        return None

    prepared = _prepare_offline_stanza_dir(sentencizer)
    if prepared is None:
        return None
    stanza_dir, lang, package = prepared
    # the bundled checkpoints predate the installed stanza, see the helper
    _ensure_legacy_stanza_tokenizers_load()

    kwargs = {
        "lang": lang,
        "dir": stanza_dir,
        # the manifest only advertises the bundled weights, so ask for them by
        # name instead of letting stanza resolve "default" to something absent
        "package": package,
        "processors": "tokenize",
        "use_gpu": False,
        "logging_level": "WARNING",
    }
    # stanza refreshes "resources_<version>.json" over HTTP unless told
    # otherwise, which is the request that fails on an isolated network.
    download_method = getattr(stanza, "DownloadMethod", None)
    for method in ("NONE", "REUSE_RESOURCES"):
        value = getattr(download_method, method, None)
        if value is not None:
            kwargs["download_method"] = value
            break

    try:
        pipeline = stanza.Pipeline(**kwargs)
    except Exception as error:
        logger.warning(
            "stanza sentence splitter is not usable (%s: %s); falling back to "
            "local sentence splitting",
            type(error).__name__,
            error,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return None
    logger.info(
        "Using stanza sentence splitter (%s/%s) from %s", lang, package, stanza_dir
    )
    return pipeline


def _patch_argos_sentence_splitter() -> None:
    """Keep argostranslate's sentence splitter away from the network.

    Newer argostranslate releases build their stanza pipeline lazily and let
    stanza fetch ``resources_<version>.json`` from raw.githubusercontent.com even
    though the models shipped inside the ``.argosmodel`` package are already on
    disk.  On an air-gapped deployment that request fails for every paragraph,
    so translation never finishes.  The lazy loader is replaced by one that only
    uses the packaged models and degrades to punctuation splitting when even
    those are unavailable.
    """
    try:
        from argostranslate import sbd
    except Exception as error:  # argostranslate is optional
        logger.debug("argostranslate sentence splitter not patched: %s", error)
        return

    sentencizer = getattr(sbd, "StanzaSentencizer", None)
    if sentencizer is None or getattr(sentencizer, "_pdf2zh_offline", False):
        return

    def lazy_pipeline(self):
        pipeline = getattr(self, "_pdf2zh_offline_pipeline", None)
        if pipeline is None:
            pipeline = _build_offline_stanza_pipeline(self) or _LocalSentenceSplitter()
            self._pdf2zh_offline_pipeline = pipeline
        return pipeline

    sentencizer.lazy_pipeline = lazy_pipeline
    sentencizer._pdf2zh_offline = True


class ArgosTranslator(BaseTranslator):
    name = "argos"

    def __init__(self, lang_in, lang_out, model, ignore_cache=False, **kwargs):
        try:
            import argostranslate.package
            import argostranslate.translate
        except ImportError:
            logger.warning(
                "argos-translate is not installed, if you want to use argostranslate, please install it. If you don't use argostranslate translator, you can safely ignore this warning."
            )
            raise
        _patch_argos_sentence_splitter()
        super().__init__(lang_in, lang_out, model, ignore_cache)
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out
        self._provision_model(argostranslate, self.lang_in, self.lang_out)

    @staticmethod
    def _is_installed(argostranslate, lang_in: str, lang_out: str) -> bool:
        """Whether the pair already lives in argos' package directory."""
        try:
            for package in argostranslate.package.get_installed_packages():
                if (
                    getattr(package, "from_code", None) == lang_in
                    and getattr(package, "to_code", None) == lang_out
                ):
                    return True
        except Exception as e:
            # A broken package directory must not prevent start-up.
            logger.debug("Could not enumerate installed argos packages: %s", e)
        return False

    def _provision_model(self, argostranslate, lang_in: str, lang_out: str):
        """Install the pair from the offline bundle, else from the remote index."""
        bundled = offline_models.ensure_argos_model(lang_in, lang_out)
        if bundled is not None:
            if self._is_installed(argostranslate, lang_in, lang_out):
                return
            logger.info("Installing bundled argos model %s", bundled.name)
            argostranslate.package.install_from_path(bundled)
            return

        # No bundle available (plain pip install): keep the previous online path.
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        try:
            available_package = list(
                filter(
                    lambda x: x.from_code == self.lang_in
                    and x.to_code == self.lang_out,
                    available_packages,
                )
            )[0]
        except Exception:
            raise ValueError(
                "lang_in and lang_out pair not supported by Argos Translate."
            )
        download_path = available_package.download()
        argostranslate.package.install_from_path(download_path)

    def translate(self, text: str, ignore_cache: bool = False):
        # Translate
        import argostranslate.translate  # noqa: F401

        installed_languages = (
            argostranslate.translate.get_installed_languages()  # noqa: F821
        )
        from_lang = list(filter(lambda x: x.code == self.lang_in, installed_languages))[
            0
        ]
        to_lang = list(filter(lambda x: x.code == self.lang_out, installed_languages))[
            0
        ]
        translation = from_lang.get_translation(to_lang)
        translatedText = translation.translate(text)
        return translatedText


class FirefoxTranslator(BaseTranslator):
    """Local neural translation through `firefox-translations` (CTranslate2).

    Runs entirely on the CPU with the models shipped in the offline bundle
    (see :mod:`pdf2zh.offline_models`), so no network access is required.
    """

    name = "firefox"
    envs = {
        "FIREFOX_DEVICE": "cpu",
        "FIREFOX_COMPUTE_TYPE": "int8",
        "FIREFOX_INTER_THREADS": "1",
        "FIREFOX_INTRA_THREADS": "0",
        "FIREFOX_BEAM_SIZE": "1",
    }
    lang_map = {
        "zh-cn": "zh",
        "zh-hans": "zh",
        "zh-hant": "zh",
        "zh-tw": "zh",
        "en-us": "en",
        "en-gb": "en",
    }

    def __init__(
        self, lang_in, lang_out, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        try:
            from firefox_translations import Translator
        except ImportError:
            logger.warning(
                "firefox-translations is not installed, if you want to use the "
                "firefox translator, please install it. If you don't use the "
                "firefox translator, you can safely ignore this warning."
            )
            raise
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.src_lang = self.lang_in.lower()
        self.trg_lang = self.lang_out.lower()

        model_dir = offline_models.ensure_firefox_model(self.src_lang, self.trg_lang)
        logger.info("Loading firefox model from %s", model_dir)
        self.translator = Translator(
            src_lang=self.src_lang,
            trg_lang=self.trg_lang,
            model_dir=str(model_dir),
            device=(self.envs.get("FIREFOX_DEVICE") or "cpu").strip(),
            compute_type=(self.envs.get("FIREFOX_COMPUTE_TYPE") or "int8").strip(),
            inter_threads=self._int_env("FIREFOX_INTER_THREADS", 1),
            intra_threads=self._int_env("FIREFOX_INTRA_THREADS", 0),
            beam_size=self._int_env("FIREFOX_BEAM_SIZE", 1),
        )

    def _int_env(self, key: str, default: int) -> int:
        """Read an integer setting, tolerating empty or malformed values."""
        try:
            return int(self.envs.get(key, default))
        except (TypeError, ValueError):
            return default

    def do_translate(self, text: str) -> str:
        return self.translator.translate(text)

    def __del__(self):
        try:
            self.translator.unload()
        except Exception:
            pass


class GrokTranslator(OpenAITranslator):
    # https://docs.x.ai/docs/overview#getting-started
    name = "grok"
    envs = {
        "GROK_API_KEY": None,
        "GROK_MODEL": "grok-2-1212",
        "GROK_BASE_URL": "https://api.x.ai/v1",  # Configurable base URL
        "GROK_STREAM": "true",  # Configurable: set to "true" (default) or "false"
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = self.envs.get("GROK_BASE_URL", "https://api.x.ai/v1")
        api_key = self.envs["GROK_API_KEY"]
        if not model:
            model = self.envs["GROK_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        # Override stream setting from config (default to True)
        stream_val = self.envs.get("GROK_STREAM", "true").lower()
        self.stream = stream_val == "true"


class GroqTranslator(OpenAITranslator):
    name = "groq"
    envs = {
        "GROQ_API_KEY": None,
        "GROQ_MODEL": "llama-3-3-70b-versatile",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://api.groq.com/openai/v1"
        api_key = self.envs["GROQ_API_KEY"]
        if not model:
            model = self.envs["GROQ_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt


class DeepseekTranslator(OpenAITranslator):
    name = "deepseek"
    envs = {
        "DEEPSEEK_API_KEY": None,
        "DEEPSEEK_MODEL": "deepseek-chat",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://api.deepseek.com/v1"
        api_key = self.envs["DEEPSEEK_API_KEY"]
        if not model:
            model = self.envs["DEEPSEEK_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt


class MiniMaxTranslator(OpenAITranslator):
    # https://platform.minimaxi.com/document/introduction
    name = "minimax"
    envs = {
        "MINIMAX_API_KEY": None,
        "MINIMAX_MODEL": "MiniMax-M2.7",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://api.minimax.io/v1"
        api_key = self.envs["MINIMAX_API_KEY"]
        if not model:
            model = self.envs["MINIMAX_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.options = {"temperature": 0.1}
        self.prompttext = prompt


class OpenAIlikedTranslator(OpenAITranslator):
    name = "openailiked"
    envs = {
        "OPENAILIKED_BASE_URL": None,
        "OPENAILIKED_API_KEY": None,
        "OPENAILIKED_MODEL": None,
        "OPENAILIKED_STREAM": "false",  # Configurable: set to "true" or "false"
        "OPENAILIKED_STOP_TOKENS": "",  # Space separated list of stop tokens
        "OPENAILIKED_MAX_TOKENS": -1,  # Specify -1 to call the API without setting max_tokens
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        if self.envs["OPENAILIKED_BASE_URL"]:
            base_url = self.envs["OPENAILIKED_BASE_URL"]
        else:
            raise ValueError("The OPENAILIKED_BASE_URL is missing.")
        if not model:
            if self.envs["OPENAILIKED_MODEL"]:
                model = self.envs["OPENAILIKED_MODEL"]
            else:
                raise ValueError("The OPENAILIKED_MODEL is missing.")
        if self.envs["OPENAILIKED_API_KEY"] is None:
            api_key = "openailiked"
        else:
            api_key = self.envs["OPENAILIKED_API_KEY"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
            prompt=prompt,
            stop_tokens=self.envs.get("OPENAILIKED_STOP_TOKENS", "").split(),
            max_tokens=int(self.envs.get("OPENAILIKED_MAX_TOKENS", -1)),
        )
        # Parse stream option from config (default to False for compatibility)
        stream_val = self.envs.get("OPENAILIKED_STREAM", "false").lower()
        self.stream = stream_val == "true"

    def do_translate(self, text) -> str:
        """Override to support configurable streaming."""
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text, self.prompttext),
            stream=self.stream,
        )
        if self.stream:
            collected = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    collected.append(chunk.choices[0].delta.content)
            content = "".join(collected).strip()
        else:
            if not response.choices:
                if hasattr(response, "error"):
                    raise ValueError("Error response from Service", response.error)
            content = response.choices[0].message.content.strip()
        content = self.think_filter_regex.sub("", content).strip()
        return content


class QwenMtTranslator(OpenAITranslator):
    """
    Use Qwen-MT model from Aliyun. it's designed for translating.
    Since Traditional Chinese is not yet supported by Aliyun. it will be also translated to Simplified Chinese, when it's selected.
    There's special parameters in the message to the server.
    """

    name = "qwen-mt"
    envs = {
        "ALI_MODEL": "qwen-mt-turbo",
        "ALI_API_KEY": None,
        "ALI_DOMAINS": "This sentence is extracted from a scientific paper. When translating, please pay close attention to the use of specialized troubleshooting terminologies and adhere to scientific sentence structures to maintain the technical rigor and precision of the original text.",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = self.envs["ALI_API_KEY"]

        if not model:
            model = self.envs["ALI_MODEL"]

        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt

    @staticmethod
    def lang_mapping(input_lang: str) -> str:
        """
        Mapping the language code to the language code that Aliyun Qwen-Mt model supports.
        Since all existings languagues codes used in gui.py are able to be mapped, the original
        languague code will not be checked.
        """
        langdict = {
            "zh": "Chinese",
            "zh-TW": "Chinese",
            "en": "English",
            "fr": "French",
            "de": "German",
            "ja": "Japanese",
            "ko": "Korean",
            "ru": "Russian",
            "es": "Spanish",
            "it": "Italian",
        }

        return langdict[input_lang]

    def do_translate(self, text) -> str:
        """
        Qwen-MT Model reqeust to send translation_options to the server.
        domains are options, but suggested. it must be in English.
        """
        translation_options = {
            "source_lang": self.lang_mapping(self.lang_in),
            "target_lang": self.lang_mapping(self.lang_out),
            "domains": self.envs["ALI_DOMAINS"],
        }
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=[{"role": "user", "content": text}],
            extra_body={"translation_options": translation_options},
        )
        return response.choices[0].message.content.strip()

import asyncio
import cgi
import os
import shutil
import uuid
from asyncio import CancelledError
from pathlib import Path
import typing as T

import gradio as gr
import requests
import tqdm
from gradio_pdf import PDF
from string import Template
import logging

from pdf2zh import __version__
from pdf2zh.high_level import translate
from pdf2zh.offline_assets import prepare_offline_gui_assets
from pdf2zh.doclayout import ModelInstance
from pdf2zh.config import ConfigManager
from pdf2zh.translator import (
    AnythingLLMTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DifyTranslator,
    ArgosTranslator,
    FirefoxTranslator,
    GeminiTranslator,
    GoogleTranslator,
    MiniMaxTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAITranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
    GrokTranslator,
    GroqTranslator,
    DeepseekTranslator,
    OpenAIlikedTranslator,
    QwenMtTranslator,
    X302AITranslator,
)
from babeldoc.docvision.doclayout import OnnxModel
from babeldoc import __version__ as babeldoc_version

logger = logging.getLogger(__name__)


# The interface is used by Chinese speaking users on an intranet deployment, so
# labels are translated by default.  English stays the lookup key, which means
# setting ``PDF2ZH_GUI_LANG=en`` in the config (or the environment) restores the
# original wording without touching this file.
_GUI_LANGUAGE = str(ConfigManager.get("PDF2ZH_GUI_LANG", "zh") or "zh").strip().lower()

_ZH_TEXT = {
    "PDFMathTranslate - PDF Translation with preserved formats": (
        "PDFMathTranslate - 保留排版的 PDF 翻译"
    ),
    "File": "文件",
    "Link": "链接",
    "Type": "输入方式",
    "Option": "选项",
    "Service": "翻译引擎",
    "Translate from": "源语言",
    "Translate to": "目标语言",
    "Pages": "页码范围",
    "All": "全部",
    "First": "第一页",
    "First 5 pages": "前 5 页",
    "First 20 pages": "前 20 页",
    "Others": "自定义",
    "Page range": "页码范围（例：1-3,5）",
    "Open for More Experimental Options!": "更多实验性选项",
    "Experimental": "实验性功能",
    "number of threads": "线程数",
    "Skip font subsetting": "跳过字体子集化",
    "Ignore cache": "忽略缓存",
    "Custom formula font regex (vfont)": "自定义公式字体正则 (vfont)",
    "Custom Prompt for llm": "自定义 LLM 提示词",
    "Translation Mode": "翻译模式",
    "Enable BabelDOC experimental backend": "启用 BabelDOC 实验性后端",
    "fast": "快速（内置 v1 内核）",
    "precise": "精准（v2 内核，需额外安装）",
    "Translated": "翻译结果",
    "Download Translation (Mono)": "下载译文（单语）",
    "Download Translation (Dual)": "下载译文（双语对照）",
    "Translate": "开始翻译",
    "Cancel": "取消",
    "Preview": "预览",
    "Document Preview": "文档预览",
    "Technical details": "技术信息",
    "Simplified Chinese": "简体中文",
    "Traditional Chinese": "繁体中文",
    "English": "英语",
    "French": "法语",
    "German": "德语",
    "Japanese": "日语",
    "Korean": "韩语",
    "Russian": "俄语",
    "Spanish": "西班牙语",
    "Italian": "意大利语",
    "Argos Translate": "Argos 离线翻译",
    "Firefox Translations": "Firefox 离线翻译",
    "Ollama": "Ollama（本地模型）",
    "Xinference": "Xinference（本地模型）",
    "OpenAI-liked": "OpenAI 兼容接口（llama.cpp/vLLM 等）",
    (
        "Precise mode needs the separately installed v2 kernel and is "
        "unavailable here."
    ): "精准模式需要额外安装 v2 内核，当前不可用",
    (
        "Renders the layout more faithfully and merges paragraphs: "
        "better quality, slower."
    ): "版面还原更完整、会合并段落，译文质量更好但速度较慢",
}


def _t(text: str) -> str:
    """Translate a label for the configured interface language."""
    if _GUI_LANGUAGE.startswith("en"):
        return text
    return _ZH_TEXT.get(text, text)


def _labeled(choices) -> list:
    """Turn ``["Fast"]`` into Gradio ``[(label, value)]`` choice pairs."""
    return [(_t(choice), choice) for choice in choices]


class _LazyModel:
    """Defers model loading until first access so the GUI starts instantly."""

    def __init__(self):
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model = OnnxModel.load_available()

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        self._ensure_loaded()
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        self._ensure_loaded()
        return self._model(*args, **kwargs)


BABELDOC_MODEL = _LazyModel()
# The following variables associate strings with translators
service_map: dict[str, BaseTranslator] = {
    "Google": GoogleTranslator,
    "Bing": BingTranslator,
    "DeepL": DeepLTranslator,
    "DeepLX": DeepLXTranslator,
    "Ollama": OllamaTranslator,
    "Xinference": XinferenceTranslator,
    "AzureOpenAI": AzureOpenAITranslator,
    "OpenAI": OpenAITranslator,
    "Zhipu": ZhipuTranslator,
    "ModelScope": ModelScopeTranslator,
    "Silicon": SiliconTranslator,
    "Gemini": GeminiTranslator,
    "Azure": AzureTranslator,
    "Tencent": TencentTranslator,
    "Dify": DifyTranslator,
    "AnythingLLM": AnythingLLMTranslator,
    "Argos Translate": ArgosTranslator,
    "Firefox Translations": FirefoxTranslator,
    "Grok": GrokTranslator,
    "Groq": GroqTranslator,
    "DeepSeek": DeepseekTranslator,
    "MiniMax": MiniMaxTranslator,
    "OpenAI-liked": OpenAIlikedTranslator,
    "Ali Qwen-Translation": QwenMtTranslator,
    "302.AI": X302AITranslator,
}

# The following variables associate strings with specific languages
lang_map = {
    "Simplified Chinese": "zh",
    "Traditional Chinese": "zh-TW",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Spanish": "es",
    "Italian": "it",
}

# The following variable associate strings with page ranges
page_map = {
    "All": None,
    "First": [0],
    "First 5 pages": list(range(0, 5)),
    "Others": None,
}

# Check if this is a public demo, which has resource limits
flag_demo = False

# Limit resources
if ConfigManager.get("PDF2ZH_DEMO"):
    flag_demo = True
    service_map = {
        "Google": GoogleTranslator,
    }
    page_map = {
        "First": [0],
        "First 20 pages": list(range(0, 20)),
    }
    client_key = ConfigManager.get("PDF2ZH_CLIENT_KEY")
    server_key = ConfigManager.get("PDF2ZH_SERVER_KEY")


# Limit Enabled Services
enabled_services: T.Optional[T.List[str]] = ConfigManager.get("ENABLED_SERVICES")
if isinstance(enabled_services, list):
    # Services always offered in addition to the allow-list. Set this to an empty
    # array (or remove the entry) in config to restrict the UI to exactly the
    # allow-list - useful for an offline/intranet deployment where the cloud
    # engines should never show up.
    default_services = ConfigManager.get("DEFAULT_SERVICES", ["Google", "Bing"])
    if isinstance(default_services, str):
        default_services = [default_services]
    enabled_services_names = [str(_).lower().strip() for _ in enabled_services]
    enabled_services = [
        k
        for k in service_map.keys()
        if str(k).lower().strip() in enabled_services_names
    ]
    if len(enabled_services) == 0:
        raise RuntimeError("No services available.")
    enabled_services = list(default_services) + enabled_services
else:
    enabled_services = list(service_map.keys())


# Translation kernels actually present on this machine.  "precise" needs the
# pdf2zh_next submodule plus a virtual environment of its own, so it is only
# offered when both exist - selecting an unprepared kernel used to raise
# "submodule not found" in the middle of a translation.
def _available_modes() -> list[str]:
    from pdf2zh.kernel import KernelRegistry

    modes = [
        name for name in ("fast", "precise") if name in set(KernelRegistry.available())
    ]
    return modes or ["fast"]


available_modes = _available_modes()


# Configure about Gradio show keys
hidden_gradio_details: bool = bool(ConfigManager.get("HIDDEN_GRADIO_DETAILS"))


# Public demo control
def verify_recaptcha(response):
    """
    This function verifies the reCAPTCHA response.
    """
    recaptcha_url = "https://www.google.com/recaptcha/api/siteverify"
    data = {"secret": server_key, "response": response}
    result = requests.post(recaptcha_url, data=data).json()
    return result.get("success")


def download_with_limit(url: str, save_path: str, size_limit: int) -> str:
    """
    This function downloads a file from a URL and saves it to a specified path.

    Inputs:
        - url: The URL to download the file from
        - save_path: The path to save the file to
        - size_limit: The maximum size of the file to download

    Returns:
        - The path of the downloaded file
    """
    chunk_size = 1024
    total_size = 0
    with requests.get(url, stream=True, timeout=10) as response:
        response.raise_for_status()
        content = response.headers.get("Content-Disposition")
        try:  # filename from header
            _, params = cgi.parse_header(content)
            filename = params["filename"]
        except Exception:  # filename from url
            filename = os.path.basename(url)
        filename = os.path.splitext(os.path.basename(filename))[0] + ".pdf"
        with open(save_path / filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                total_size += len(chunk)
                if size_limit and total_size > size_limit:
                    raise gr.Error("Exceeds file size limit")
                file.write(chunk)
    return save_path / filename


def stop_translate_file(state: dict) -> None:
    """
    This function stops the translation process.

    Inputs:
        - state: The state of the translation process

    Returns:- None
    """
    session_id = state["session_id"]
    if session_id is None:
        return
    if session_id in cancellation_event_map:
        logger.info(f"Stopping translation for session {session_id}")
        cancellation_event_map[session_id].set()


def translate_file(
    file_type,
    file_input,
    link_input,
    service,
    lang_from,
    lang_to,
    page_range,
    page_input,
    prompt,
    threads,
    skip_subset_fonts,
    ignore_cache,
    vfont,
    mode_choice,
    babeldoc_backend,
    recaptcha_response,
    state,
    progress=gr.Progress(),
    *envs,
):
    """
    This function translates a PDF file from one language to another.

    Inputs:
        - file_type: The type of file to translate
        - file_input: The file to translate
        - link_input: The link to the file to translate
        - service: The translation service to use
        - lang_from: The language to translate from
        - lang_to: The language to translate to
        - page_range: The range of pages to translate
        - page_input: The input for the page range
        - prompt: The custom prompt for the llm
        - threads: The number of threads to use
        - mode_choice: The selected translation kernel ("fast" or "precise")
        - babeldoc_backend: Whether to translate with the BabelDOC backend
        - recaptcha_response: The reCAPTCHA response
        - state: The state of the translation process
        - progress: The progress bar
        - envs: The environment variables

    Returns:
        - The translated file
        - The translated file
        - The translated file
        - The progress bar
        - The progress bar
        - The progress bar
    """
    session_id = uuid.uuid4()
    state["session_id"] = session_id
    cancellation_event_map[session_id] = asyncio.Event()
    # Translate PDF content using selected service.
    if flag_demo and not verify_recaptcha(recaptcha_response):
        raise gr.Error("reCAPTCHA fail")

    progress(0, desc="Starting translation...")

    output = Path("pdf2zh_files")
    output.mkdir(parents=True, exist_ok=True)

    if file_type == "File":
        if not file_input:
            raise gr.Error("No input")
        file_path = shutil.copy(file_input, output)
    else:
        if not link_input:
            raise gr.Error("No input")
        file_path = download_with_limit(
            link_input,
            output,
            5 * 1024 * 1024 if flag_demo else None,
        )

    filename = os.path.splitext(os.path.basename(file_path))[0]
    file_raw = output / f"{filename}.pdf"
    file_mono = output / f"{filename}-mono.pdf"
    file_dual = output / f"{filename}-dual.pdf"

    translator = service_map[service]
    if page_range != "Others":
        selected_page = page_map[page_range]
    else:
        selected_page = []
        for p in page_input.split(","):
            if "-" in p:
                start, end = p.split("-")
                selected_page.extend(range(int(start) - 1, int(end)))
            else:
                selected_page.append(int(p) - 1)
    lang_from = lang_map[lang_from]
    lang_to = lang_map[lang_to]

    _envs = {}
    for i, env in enumerate(translator.envs.items()):
        _envs[env[0]] = envs[i]
    for k, v in _envs.items():
        if str(k).upper().endswith("API_KEY") and str(v) == "***":
            # Load Real API_KEYs from local configure file
            real_keys: str = ConfigManager.get_env_by_translatername(
                translator, k, None
            )
            _envs[k] = real_keys

    print(f"Files before translation: {os.listdir(output)}")

    def progress_bar(t: tqdm.tqdm):
        desc = getattr(t, "desc", "Translating...")
        if desc == "":
            desc = "Translating..."
        progress(t.n / t.total, desc=desc)

    try:
        threads = int(threads)
    except ValueError:
        threads = 1

    param = {
        "files": [str(file_raw)],
        "pages": selected_page,
        "lang_in": lang_from,
        "lang_out": lang_to,
        "service": f"{translator.name}",
        "output": output,
        "thread": int(threads),
        "callback": progress_bar,
        "cancellation_event": cancellation_event_map[session_id],
        "envs": _envs,
        "prompt": Template(prompt) if prompt else None,
        "skip_subset_fonts": skip_subset_fonts,
        "ignore_cache": ignore_cache,
        "vfont": vfont,  # 添加自定义公式字体正则表达式
        "model": ModelInstance.value,
    }

    try:
        if babeldoc_backend:
            # The experimental backend has its own pipeline, so it bypasses the
            # kernel registry (which only routes the built-in v1 kernels).
            babeldoc_result = babeldoc_translate_file(
                files=[str(file_raw)],
                output=str(output),
                pages=selected_page,
                lang_in=lang_from,
                lang_out=lang_to,
                service=f"{translator.name}",
                thread=int(threads),
                envs=_envs,
                prompt=str(prompt) if prompt else None,
                skip_subset_fonts=skip_subset_fonts,
                ignore_cache=ignore_cache,
                vfont=vfont,
                callback=progress_bar,
                cancellation_event=cancellation_event_map[session_id],
            )
            print(f"Files after translation: {os.listdir(output)}")
            progress(1.0, desc="Translation complete!")
            return babeldoc_result

        from pdf2zh.kernel import KernelRegistry
        from pdf2zh.kernel.protocol import TranslateRequest

        KernelRegistry.switch(mode_choice)
        kernel = KernelRegistry.get()
        request = TranslateRequest(
            files=[str(file_raw)],
            output=str(output),
            pages=selected_page,
            lang_in=lang_from,
            lang_out=lang_to,
            service=f"{translator.name}",
            thread=int(threads),
            envs=_envs,
            prompt=str(prompt) if prompt else None,
            skip_subset_fonts=skip_subset_fonts,
            ignore_cache=ignore_cache,
            vfont=vfont,
        )
        kernel.translate(
            request,
            callback=progress_bar,
            cancellation_event=cancellation_event_map[session_id],
        )
    except CancelledError:
        del cancellation_event_map[session_id]
        raise gr.Error("Translation cancelled")
    print(f"Files after translation: {os.listdir(output)}")

    if not file_mono.exists() or not file_dual.exists():
        raise gr.Error("No output")

    progress(1.0, desc="Translation complete!")

    return (
        str(file_mono),
        str(file_mono),
        str(file_dual),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )


def babeldoc_translate_file(**kwargs):
    from babeldoc.high_level import init as babeldoc_init

    babeldoc_init()
    from babeldoc.high_level import async_translate as babeldoc_translate
    from babeldoc.translation_config import TranslationConfig as YadtConfig

    for translator in [
        GoogleTranslator,
        BingTranslator,
        DeepLTranslator,
        DeepLXTranslator,
        OllamaTranslator,
        XinferenceTranslator,
        AzureOpenAITranslator,
        OpenAITranslator,
        ZhipuTranslator,
        ModelScopeTranslator,
        SiliconTranslator,
        GeminiTranslator,
        AzureTranslator,
        TencentTranslator,
        DifyTranslator,
        AnythingLLMTranslator,
        ArgosTranslator,
        FirefoxTranslator,
        GrokTranslator,
        GroqTranslator,
        DeepseekTranslator,
        OpenAIlikedTranslator,
        QwenMtTranslator,
        X302AITranslator,
    ]:
        if kwargs["service"] == translator.name:
            translator = translator(
                kwargs["lang_in"],
                kwargs["lang_out"],
                "",
                envs=kwargs["envs"],
                prompt=kwargs["prompt"],
                ignore_cache=kwargs["ignore_cache"],
            )
            break
    else:
        raise ValueError("Unsupported translation service")
    import asyncio
    from babeldoc.main import create_progress_handler

    # ``pages`` arrives as the 0-based page list the GUI selected, whereas
    # babeldoc expects a 1-based range string such as "1-3,5".
    selected_pages = kwargs.get("pages") or []
    if isinstance(selected_pages, str):
        pages = selected_pages
    else:
        pages = ",".join(str(int(page) + 1) for page in selected_pages)
    output_dir = Path(kwargs["output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in kwargs["files"]:
        file = file.strip("\"'")
        yadt_config = YadtConfig(
            input_file=file,
            font=None,
            pages=pages or None,
            output_dir=str(output_dir),
            doc_layout_model=BABELDOC_MODEL,
            translator=translator,
            debug=False,
            lang_in=kwargs["lang_in"],
            lang_out=kwargs["lang_out"],
            no_dual=False,
            no_mono=False,
            qps=kwargs["thread"],
            use_rich_pbar=False,
            disable_rich_text_translate=not isinstance(translator, OpenAITranslator),
            formular_font_pattern=kwargs.get("vfont") or None,
            skip_clean=kwargs["skip_subset_fonts"],
            report_interval=0.5,
        )

        async def yadt_translate_coro(yadt_config):
            progress_context, progress_handler = create_progress_handler(yadt_config)
            file_mono = None
            file_dual = None

            # 开始翻译
            with progress_context:
                async for event in babeldoc_translate(yadt_config):
                    progress_handler(event)
                    if yadt_config.debug:
                        logger.debug(event)
                    # with use_rich_pbar=False the progress context is a tqdm
                    # bar, which is exactly what the Gradio callback expects
                    if kwargs.get("callback"):
                        kwargs["callback"](progress_context)
                    if kwargs["cancellation_event"].is_set():
                        yadt_config.cancel_translation()
                        raise CancelledError
                    if event["type"] == "finish":
                        result = event["translate_result"]
                        logger.info("Translation Result:")
                        logger.info(f"  Original PDF: {result.original_pdf_path}")
                        logger.info(f"  Time Cost: {result.total_seconds:.2f}s")
                        logger.info(f"  Mono PDF: {result.mono_pdf_path or 'None'}")
                        logger.info(f"  Dual PDF: {result.dual_pdf_path or 'None'}")
                        file_mono = result.mono_pdf_path
                        file_dual = result.dual_pdf_path
                        break
            import gc

            gc.collect()
            if not file_mono or not file_dual:
                raise gr.Error("BabelDOC produced no output")
            return (
                str(file_mono),
                str(file_mono),
                str(file_dual),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )

        return asyncio.run(yadt_translate_coro(yadt_config))


# Global setup
custom_blue = gr.themes.Color(
    c50="#E8F3FF",
    c100="#BEDAFF",
    c200="#94BFFF",
    c300="#6AA1FF",
    c400="#4080FF",
    c500="#165DFF",  # Primary color
    c600="#0E42D2",
    c700="#0A2BA6",
    c800="#061D79",
    c900="#03114D",
    c950="#020B33",
)

# Font stacks used by the GUI. Plain strings are treated as *local* fonts by
# Gradio; using a GoogleFont here would make the page request
# https://fonts.googleapis.com, which never resolves in an intranet.
system_fonts = (
    "ui-sans-serif",
    "system-ui",
    "-apple-system",
    "Segoe UI",
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "sans-serif",
)
system_mono_fonts = (
    "ui-monospace",
    "Consolas",
    "Menlo",
    "monospace",
)

custom_css = """
    .secondary-text {color: #999 !important;}
    footer {visibility: hidden}
    .env-warning {color: #dd5500 !important;}
    .env-success {color: #559900 !important;}

    /* Add dashed border to input-file class */
    .input-file {
        border: 1.2px dashed #165DFF !important;
        border-radius: 6px !important;
    }

    .progress-bar-wrap {
        border-radius: 8px !important;
    }

    .progress-bar {
        border-radius: 8px !important;
    }

    .pdf-canvas canvas {
        width: 100%;
    }
    """

demo_recaptcha = """
    <script src="https://www.google.com/recaptcha/api.js?render=explicit" async defer></script>
    <script type="text/javascript">
        var onVerify = function(token) {
            el=document.getElementById('verify').getElementsByTagName('textarea')[0];
            el.value=token;
            el.dispatchEvent(new Event('input'));
        };
    </script>
    """

tech_details_string = f"""
                    <summary>{_t("Technical details")}</summary>
                    - GitHub: <a href="https://github.com/Byaidu/PDFMathTranslate">Byaidu/PDFMathTranslate</a><br>
                    - BabelDOC: <a href="https://github.com/funstory-ai/BabelDOC">funstory-ai/BabelDOC</a><br>
                    - GUI by: <a href="https://github.com/reycn">Rongxin</a><br>
                    - pdf2zh Version: {__version__} <br>
                    - BabelDOC Version: {babeldoc_version}
                """
cancellation_event_map = {}


# The following code creates the GUI
with gr.Blocks(
    title=_t("PDFMathTranslate - PDF Translation with preserved formats"),
    theme=gr.themes.Default(
        primary_hue=custom_blue,
        spacing_size="md",
        radius_size="lg",
        font=system_fonts,
        font_mono=system_mono_fonts,
    ),
    css=custom_css,
    head=demo_recaptcha if flag_demo else "",
) as demo:
    gr.Markdown(
        "# [PDFMathTranslate @ GitHub](https://github.com/Byaidu/PDFMathTranslate)"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## " + _t("File") + (" | < 5 MB" if flag_demo else ""))
            file_type = gr.Radio(
                choices=_labeled(["File", "Link"]),
                label=_t("Type"),
                value="File",
            )
            file_input = gr.File(
                label=_t("File"),
                file_count="single",
                file_types=[".pdf", ".doc", ".docx"],
                type="filepath",
                elem_classes=["input-file"],
            )
            link_input = gr.Textbox(
                label=_t("Link"),
                visible=False,
                interactive=True,
            )
            gr.Markdown("## " + _t("Option"))
            service = gr.Dropdown(
                label=_t("Service"),
                choices=_labeled(enabled_services),
                value=enabled_services[0],
            )
            # Each engine declares its own env entries and the count differs
            # (OpenAI-liked/OpenAI use six, Firefox five, Google/Bing none), so
            # the number of boxes must follow the registry.  The last component
            # of ``envs`` is always the custom prompt.
            env_slots = max(
                (len(getattr(t, "envs", {}) or {}) for t in service_map.values()),
                default=0,
            )
            env_slots = max(3, env_slots)
            envs = []
            for i in range(env_slots):
                envs.append(
                    gr.Textbox(
                        visible=False,
                        interactive=True,
                    )
                )
            with gr.Row():
                lang_from = gr.Dropdown(
                    label=_t("Translate from"),
                    choices=_labeled(lang_map.keys()),
                    value=ConfigManager.get("PDF2ZH_LANG_FROM", "English"),
                )
                lang_to = gr.Dropdown(
                    label=_t("Translate to"),
                    choices=_labeled(lang_map.keys()),
                    value=ConfigManager.get("PDF2ZH_LANG_TO", "Simplified Chinese"),
                )
            page_range = gr.Radio(
                choices=_labeled(page_map.keys()),
                label=_t("Pages"),
                value=list(page_map.keys())[0],
            )

            page_input = gr.Textbox(
                label=_t("Page range"),
                visible=False,
                interactive=True,
            )

            with gr.Accordion(_t("Open for More Experimental Options!"), open=False):
                gr.Markdown("#### " + _t("Experimental"))
                threads = gr.Textbox(
                    label=_t("number of threads"), interactive=True, value="4"
                )
                skip_subset_fonts = gr.Checkbox(
                    label=_t("Skip font subsetting"), interactive=True, value=False
                )
                ignore_cache = gr.Checkbox(
                    label=_t("Ignore cache"), interactive=True, value=False
                )
                vfont = gr.Textbox(
                    label=_t("Custom formula font regex (vfont)"),
                    interactive=True,
                    value=ConfigManager.get("PDF2ZH_VFONT", ""),
                )
                prompt = gr.Textbox(
                    label=_t("Custom Prompt for llm"), interactive=True, visible=False
                )
                mode_choice = gr.Dropdown(
                    label=_t("Translation Mode"),
                    choices=[(_t(mode), mode) for mode in available_modes],
                    value=available_modes[0],
                    interactive=True,
                    info=(
                        ""
                        if "precise" in available_modes
                        else _t(
                            "Precise mode needs the separately installed v2 "
                            "kernel and is unavailable here."
                        )
                    ),
                )
                babeldoc_backend = gr.Checkbox(
                    label=_t("Enable BabelDOC experimental backend"),
                    interactive=True,
                    value=False,
                    info=_t(
                        "Renders the layout more faithfully and merges "
                        "paragraphs: better quality, slower."
                    ),
                )
                envs.append(prompt)

            def on_select_service(service, evt: gr.EventData):
                translator = service_map[service]
                # one hidden update per component, prompt slot included
                _envs = [gr.update(visible=False, value="") for _ in range(len(envs))]
                for i, env in enumerate(translator.envs.items()):
                    label = env[0]
                    value = ConfigManager.get_env_by_translatername(
                        translator, env[0], env[1]
                    )
                    visible = True
                    if hidden_gradio_details:
                        if (
                            "MODEL" not in str(label).upper()
                            and value
                            and hidden_gradio_details
                        ):
                            visible = False
                        # Hidden Keys From Gradio
                        if "API_KEY" in label.upper():
                            value = "***"  # We use "***" Present Real API_KEY
                    _envs[i] = gr.update(
                        visible=visible,
                        label=label,
                        value=value,
                    )
                _envs[-1] = gr.update(visible=translator.CustomPrompt)
                return _envs

            def on_select_filetype(file_type):
                return (
                    gr.update(visible=file_type == "File"),
                    gr.update(visible=file_type == "Link"),
                )

            def on_select_page(choice):
                if choice == "Others":
                    return gr.update(visible=True)
                else:
                    return gr.update(visible=False)

            def on_vfont_change(value):
                ConfigManager.set("PDF2ZH_VFONT", value)
                return value

            output_title = gr.Markdown("## " + _t("Translated"), visible=False)
            output_file_mono = gr.File(
                label=_t("Download Translation (Mono)"), visible=False
            )
            output_file_dual = gr.File(
                label=_t("Download Translation (Dual)"), visible=False
            )
            recaptcha_response = gr.Textbox(
                label="reCAPTCHA Response", elem_id="verify", visible=False
            )
            recaptcha_box = gr.HTML('<div id="recaptcha-box"></div>')
            translate_btn = gr.Button(_t("Translate"), variant="primary")
            cancellation_btn = gr.Button(_t("Cancel"), variant="secondary")
            tech_details_tog = gr.Markdown(
                tech_details_string,
                elem_classes=["secondary-text"],
            )
            page_range.select(on_select_page, page_range, page_input)
            service.select(
                on_select_service,
                service,
                envs,
            )
            vfont.change(on_vfont_change, inputs=vfont, outputs=None)
            file_type.select(
                on_select_filetype,
                file_type,
                [file_input, link_input],
                js=(
                    f"""
                    (a,b)=>{{
                        try{{
                            grecaptcha.render('recaptcha-box',{{
                                'sitekey':'{client_key}',
                                'callback':'onVerify'
                            }});
                        }}catch(error){{}}
                        return [a];
                    }}
                    """
                    if flag_demo
                    else ""
                ),
            )

        with gr.Column(scale=2):
            gr.Markdown("## " + _t("Preview"))
            preview = PDF(label=_t("Document Preview"), visible=True, height=2000)

    # Event handlers
    file_input.upload(
        lambda x: x,
        inputs=file_input,
        outputs=preview,
        js=(
            f"""
            (a,b)=>{{
                try{{
                    grecaptcha.render('recaptcha-box',{{
                        'sitekey':'{client_key}',
                        'callback':'onVerify'
                    }});
                }}catch(error){{}}
                return [a];
            }}
            """
            if flag_demo
            else ""
        ),
    )

    state = gr.State({"session_id": None})

    translate_btn.click(
        translate_file,
        inputs=[
            file_type,
            file_input,
            link_input,
            service,
            lang_from,
            lang_to,
            page_range,
            page_input,
            prompt,
            threads,
            skip_subset_fonts,
            ignore_cache,
            vfont,
            mode_choice,
            babeldoc_backend,
            recaptcha_response,
            state,
            *envs,
        ],
        outputs=[
            output_file_mono,
            preview,
            output_file_dual,
            output_file_mono,
            output_file_dual,
            output_title,
        ],
    ).then(lambda: None, js="()=>{grecaptcha.reset()}" if flag_demo else "")

    cancellation_btn.click(
        stop_translate_file,
        inputs=[state],
    )


def parse_user_passwd(file_path: str) -> tuple:
    """
    Parse the user name and password from the file.

    Inputs:
        - file_path: The file path to read.
    Outputs:
        - tuple_list: The list of tuples of user name and password.
        - content: The content of the file
    """
    tuple_list = []
    content = ""
    if not file_path:
        return tuple_list, content
    if len(file_path) == 2:
        try:
            with open(file_path[1], "r", encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Error: File '{file_path[1]}' not found.")
    try:
        with open(file_path[0], "r", encoding="utf-8") as file:
            tuple_list = [
                tuple(line.strip().split(",")) for line in file if line.strip()
            ]
    except FileNotFoundError:
        print(f"Error: File '{file_path[0]}' not found.")
    return tuple_list, content


def setup_gui(
    share: bool = False, auth_file: list = ["", ""], server_port=7860
) -> None:
    """
    Setup the GUI with the given parameters.

    Inputs:
        - share: Whether to share the GUI.
        - auth_file: The file path to read the user name and password.

    Outputs:
        - None
    """
    # Gradio's launch-time localhost probe uses httpx, which honors
    # HTTP(S)_PROXY env vars. With global-mode proxy software the probe
    # to 127.0.0.1 is routed to the proxy and fails, so exclude loopback
    # addresses from proxying before launching.
    for var in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(var, "")
        tokens = {t.strip() for t in existing.split(",") if t.strip()}
        tokens.update(("127.0.0.1", "localhost", "::1"))
        os.environ[var] = ",".join(sorted(tokens))

    # Serve front-end assets (e.g. the pdf.js worker) from disk instead of
    # public CDNs, and keep Gradio from calling home.
    try:
        prepare_offline_gui_assets()
    except Exception as e:  # never block the GUI on asset preparation
        logger.warning("Could not prepare offline GUI assets: %s", e)

    user_list, html = parse_user_passwd(auth_file)

    auth_kwargs = {}
    if len(user_list) > 0:
        auth_kwargs = {"auth": user_list, "auth_message": html}

    if flag_demo:
        demo.launch(server_name="0.0.0.0", max_file_size="5mb", inbrowser=True)
        return

    # Try binding addresses in order. NOTE: do NOT use "[::]" here.
    # Gradio 5.x builds its launch-time startup probe URL from the server
    # name (http://[::]:port), and connecting to the *unspecified* IPv6
    # address "::" always fails (WinError 10049 on Windows). "0.0.0.0"
    # probes via "localhost" and "127.0.0.1" probes via itself — both work.
    bind_addresses = ["0.0.0.0", "127.0.0.1"]

    for addr in bind_addresses:
        try:
            demo.launch(
                server_name=addr,
                debug=True,
                inbrowser=True,
                share=share,
                server_port=server_port,
                **auth_kwargs,
            )
            return
        except Exception:
            print(
                f"Error launching GUI using {addr}.\n"
                "This may be caused by global mode of proxy software."
            )
            # A failed launch() leaves gradio with is_running=True and a
            # cached local_url; retrying without close() would reuse the
            # broken URL ("Rerunning server...") and fail identically.
            try:
                demo.close()
            except Exception:
                pass

    # All bind attempts failed. Do NOT fall back to share=True silently:
    # it publishes the GUI on a public *.gradio.live link (privacy risk).
    raise RuntimeError(
        "Could not bind the Gradio GUI to any local address.\n"
        "Check whether the port is already in use, or whether proxy\n"
        "software intercepts localhost connections (add 127.0.0.1/localhost\n"
        "to the proxy bypass list or set NO_PROXY), then restart."
    )


# For auto-reloading while developing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    setup_gui()

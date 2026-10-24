"""Qwen3.5 conditioning: unpadded encoding, caption sidecars and prompt cache."""
from collections import OrderedDict
from dataclasses import dataclass
import gc
import hashlib
from pathlib import Path
import logging
import torch
from ...text_cache import TextCacheStore, TextCacheEntry
from modeling.moss_image.weights import checkpoint_files
from .loader import load_text_model

logger = logging.getLogger(__name__)
TEXT_FINGERPRINT = 'moss-qwen35-9b-last4096-dequant-v2'


@dataclass
class TextCondition:
    context: torch.Tensor
    attention_mask: torch.Tensor


class MossTextStack:
    def __init__(self, path, device, *, quantization='none', offload_layers=0,
                 max_length=4096, tokenizer_path='', cache_enabled=True, tokenizer=None, model_loader=None):
        self.path = Path(path)
        self.device = torch.device(device)
        self.quantization = quantization
        self.offload_layers = offload_layers
        self.max_length = max_length
        self.cache_enabled = cache_enabled
        self._model = None
        self._loader = model_loader or load_text_model
        root = self.path if self.path.is_dir() else self.path.parent
        candidates = [root, root / 'tokenizer', root.parent / 'tokenizer']
        token_dir = Path(tokenizer_path) if tokenizer_path else next(
            (p for p in candidates if (p / 'tokenizer_config.json').is_file()), root)
        if tokenizer is None:
            from transformers import AutoTokenizer
            if tokenizer_path and not (token_dir / 'tokenizer_config.json').is_file():
                raise FileNotFoundError(f'MOSS tokenizer folder has no tokenizer_config.json: {token_dir}')
            if (token_dir / 'tokenizer_config.json').is_file():
                tokenizer = AutoTokenizer.from_pretrained(str(token_dir), local_files_only=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained('OpenMOSS-Team/MOSS-Image-Base', subfolder='tokenizer')
        self.tokenizer = tokenizer
        # Checkpoint identity and encoding policy invalidate both caption and sample caches.
        identity = '|'.join(f'{p.resolve()}:{p.stat().st_size}:{p.stat().st_mtime_ns}' for p in checkpoint_files(path))
        tokenizer_identity = getattr(tokenizer, 'name_or_path', tokenizer_path)
        for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
            token_file = token_dir / name
            if token_file.is_file():
                tokenizer_identity += hashlib.sha256(token_file.read_bytes()).hexdigest()
        fingerprint = hashlib.sha256(f'{identity}|{tokenizer_identity}|{quantization}|{max_length}'.encode()).hexdigest()[:20]
        self.store = TextCacheStore(f'{TEXT_FINGERPRINT}-{fingerprint}')
        self.entries = {}
        self.cache_root = None
        self.prompts = OrderedDict()

    @property
    def is_model_loaded(self):
        return self._model is not None

    @property
    def is_model_on_device(self):
        return self._model is not None

    def ensure_model(self):
        if self._model is None:
            self._model = self._loader(self.path, self.device, quantization=self.quantization, offload_layers=self.offload_layers)
        return self._model

    def release_model(self):
        if self._model is not None:
            offload = getattr(self._model, '_layer_offload', None)
            if offload is not None:
                offload.close()
            self._model = None
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

    offload_model = release_model

    def tokenize(self, caption):
        prompt = f'<|im_start|>user\n{caption}<|im_end|>\n<|im_start|>assistant\n<think>\n'
        return self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.max_length)['input_ids']

    def token_count(self, caption):
        return int(self.tokenize(caption).shape[1])

    @torch.no_grad()
    def _encode(self, caption):
        ids = self.tokenize(caption).to(self.device)
        model = self.ensure_model()
        with torch.autocast(self.device.type, enabled=False):
            context = model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[0]
        if context.ndim != 2 or context.shape[-1] != 4096 or not torch.isfinite(context).all():
            raise ValueError('MOSS text encoder returned invalid conditioning')
        return context.detach().cpu().contiguous()

    @staticmethod
    def _read(payload):
        context = payload.get('context') if payload else None
        if (isinstance(context, torch.Tensor) and context.ndim == 2
                and context.shape[0] > 0 and context.shape[1] == 4096):
            return context
        return None

    def _prompt(self, caption):
        if caption in self.prompts:
            self.prompts.move_to_end(caption)
            return self.prompts[caption]
        entry = (TextCacheEntry.for_image(self.cache_root / '.text-cache' / 'moss-prompts' / self.store.key(caption), caption)
                 if self.cache_enabled and self.cache_root else None)
        context = self._read(self.store.read_caption(entry)) if entry else None
        if context is None:
            context = self._encode(caption)
            if entry:
                self.store.write_caption(entry, {'context': context})
        self.prompts[caption] = context
        while len(self.prompts) > 64:
            self.prompts.popitem(last=False)
        return context

    def prepare_text_cache(self, captions, extra_prompts, *, cache_entries=(), cache_root=None):
        self.cache_root = Path(cache_root) if cache_root else None
        self.entries = {}
        for entry in cache_entries:
            self.entries.setdefault(entry.caption, []).append(entry)
        if not self.cache_enabled:
            return
        try:
            # Prompts encoded before a task supplied its cache directory must
            # also survive process restarts, without another encoder pass.
            if self.cache_root:
                for caption, context in self.prompts.items():
                    entry = TextCacheEntry.for_image(
                        self.cache_root / '.text-cache' / 'moss-prompts' / self.store.key(caption), caption)
                    if self._read(self.store.read_caption(entry)) is None:
                        self.store.write_caption(entry, {'context': context})
            for index, (caption, entries) in enumerate(self.entries.items()):
                context = None
                missing = []
                for entry in entries:
                    cached = self._read(self.store.read_caption(entry))
                    if cached is None:
                        missing.append(entry)
                    elif context is None:
                        context = cached
                if missing:
                    if context is None:
                        context = self._encode(caption)
                    for entry in missing:
                        self.store.write_caption(entry, {'context': context})
                if (index + 1) % 10 == 0:
                    logger.info('MOSS text cache: %d/%d captions', index + 1, len(self.entries))
            prompts = list(dict.fromkeys(['', *extra_prompts, *(c for c in captions if c not in self.entries)]))
            # One file per prompt bounds RAM even with thousands of DOP captions.
            for prompt in prompts:
                self._prompt(prompt)
        finally:
            self.release_model()

    def online_conditions_cached(self, captions):
        return all(c in self.prompts for c in captions)

    def precache_online_prompts(self, captions):
        unique = list(dict.fromkeys(captions))
        try:
            for caption in unique:
                self._prompt(caption)
        finally:
            self.release_model()
        return len(unique)

    def encode_text_for_batch(self, captions, *, device, dtype):
        contexts = []
        try:
            for caption in captions:
                context = None
                entries = self.entries.get(caption, []) if self.cache_enabled else []
                for entry in entries:
                    context = self._read(self.store.read_caption(entry))
                    if context is not None:
                        break
                if context is None:
                    context = self._prompt(caption)
                    for entry in entries:
                        self.store.write_caption(entry, {'context': context})
                contexts.append(context)
        finally:
            if self.cache_enabled:
                self.release_model()
        if not contexts:
            raise ValueError('MOSS text batch cannot be empty')
        length = max(c.shape[0] for c in contexts)
        result = torch.zeros(len(contexts), length, 4096, device=device, dtype=dtype)
        mask = torch.zeros(len(contexts), length, device=device, dtype=torch.bool)
        for i, context in enumerate(contexts):
            result[i, :len(context)] = context.to(device=device, dtype=dtype)
            mask[i, :len(context)] = True
        return TextCondition(result, mask)

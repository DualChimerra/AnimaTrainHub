"""Block swap -- per-layer weight swap-in/swap-out for the DiT (pushes the K2 VRAM floor down on consumer cards).

Design and measurements: ``docs/design/block-swap.md``. In one sentence: the
DiT's N transformer blocks are stacked serially and only one is computed at a
time, so keeping some of those layers' weights resident in CPU pinned memory
and moving them to VRAM only when needed trades "a fixed cost of about
8ms/block" for "about 0.8GB of VRAM per layer" (measured on krea2), pulling
K2 LoRA training's VRAM floor from 32GB down below 24GB with no accuracy cost.

This module is the **family-agnostic mechanism core** (doc SS9.2, cut 1): it
only knows about a single ``nn.ModuleList``, not krea2/anima. Wiring (which
family, which loop) happens in each family's behavior adapter layer.

Key implementation constraints (doc SS9.1, read this first):

- **Swaps ``param.data`` in place, not a rotating buffer.** LyCORIS's
  ``apply_to()`` makes the LoRA module hold a reference to the original
  Linear and wrap its forward; if the forward pass instead goes through a
  different module instance, it **completely bypasses LoRA**, and training
  silently learns nothing. So the module object itself never changes across
  the whole run -- only what ``.data`` points to changes.
- This **automatically handles fp8 correctly**: ``weight_scale`` is a
  non-persistent buffer bound to the module; since the module never changes,
  the scale always stays paired with the weight; swapping weights only swaps
  ``.data``, and the fp8 tensor moves as-is (dtype unchanged).
- The pinned master copies are **allocated once at startup**; no further
  allocation happens at runtime (allocation can only fail at startup, so it
  can fail-fast, doc SS8.1).

The forward/backward prefetch timing does not live in this module -- this
module only provides two primitives, "move a layer's weights to a GPU slot"
and "a layer is done computing, its GPU slot can be reclaimed"; loop
orchestration is left to the caller (the family's forward).

**A swapped-out layer's weights are only valid within its own forward
window.** After one pass ends, a swapped-out layer's ``param.data`` still
points at the GPU slot it used at the time, and that slot has long since been
overwritten by a later layer (double-buffer rotation: rel 0 and rel 2 share
slot 0). This is not a bug but an inevitable consequence of the design --
reading or modifying weights outside the window (export, inspection,
inference-side fp8 LoRA merge) **must call ``restore_masters()`` first**;
only the CPU pinned master copy is the persistent, complete one. There is a
dedicated test pinning down this semantics.
"""

from __future__ import annotations

import logging
from typing import Iterator

import torch
from torch import nn


logger = logging.getLogger(__name__)

_GIB = 1024 ** 3


def _pow2_ceil(n: int) -> int:
    """The smallest power of 2 >= n (n <= 1 -> 1). Uses the same rounding rule as CachingHostAllocator."""
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _pow2_decomposition(total: int) -> list[int]:
    """Decompose total into several **distinct** powers of 2 (i.e. its binary representation), descending."""
    sizes: list[int] = []
    bit = 1
    while total:
        if total & 1:
            sizes.append(bit)
        total >>= 1
        bit <<= 1
    sizes.reverse()
    return sizes


def _alloc_pinned_bytes(nbytes: int) -> torch.Tensor:
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


class PinnedAllocationError(RuntimeError):
    """Pinned (page-locked) memory allocation failed (doc SS8.1 / B6: error out, never silently degrade).

    ``cudaHostAlloc`` failures are also reported by torch as
    ``CUDA error: out of memory``, which is very easily misread as "not
    enough VRAM"; this spells out clearly that it's host page-locked memory,
    how much was being locked, and where the Windows ceiling is.
    """

    def __init__(self, nbytes: int, detail: str) -> None:
        self.nbytes = nbytes
        self.detail = detail
        super().__init__(
            f"pinned (page-locked) memory allocation failed: this attempt needed to lock "
            f"{nbytes / _GIB:.2f} GB. This is not insufficient VRAM -- swapped-out layers' "
            f"weights need to be locked in system memory, and Windows imposes a system-level "
            f"cap on lockable memory (empirically about half of physical RAM). Try lowering "
            f"blocks_to_swap, or close other memory-hungry applications and retry. "
            f"Underlying error: {detail}"
        )


class PinnedPacker:
    """Packs multiple tensors into a small number of **power-of-2-sized**
    pinned chunks, then slices out views to return.

    Why per-tensor ``pin_memory()`` doesn't work: PyTorch's host caching
    allocator rounds up **every** pinned allocation to a power of 2
    (``CachingHostAllocator.h``'s ``PowerOf2Ceil``), and the DiT's weight
    sizes happen to be particularly unlucky for this -- krea2's 16384x6144
    fp8 = 96MB rounds up to occupy 128MB, 6144x6144 = 36MB occupies 64MB,
    1536x6144 = 9MB occupies 16MB; 28 layers' worth of 11.32GB of weights
    actually locks **16.63GB** (1.47x). Windows caps ``cudaHostAlloc`` at
    roughly half of physical RAM, and a 32GB machine hits that ceiling right
    here (an actual case on a 5080 machine), while the guardrail only checks
    against the nominal 11.32GB.

    Approach: pre-allocate a handful of chunks in one shot via a binary
    decomposition of the **total byte count** (8G+2G+1G+256M+..., each chunk
    is exactly a power of 2 -> zero rounding waste from the allocator); each
    tensor is packed into some chunk by best-fit and 256B-aligned; the
    returned value is a view into that chunk (``is_pinned()`` holds, so
    ``param.data = view`` works directly, and the H2D copy proceeds as
    normal). Simulated across many family/config combinations, actual locked
    memory = weight bytes x 1.00-1.03.

    - ``total_bytes`` is the caller's precomputed **exact byte count to be
      pinned** (not an estimate: overestimating means wasted locked memory);
      passing 0 skips pre-allocation and degrades to opening chunks on demand.
    - A tensor that doesn't fit into any existing chunk (a chunk-boundary
      fragment) takes the overflow path: a chunk of exactly
      ``pow2_ceil(nbytes)`` is opened just for it -- equivalent to per-tensor
      pinning, so this is never worse than the old behavior.
    - All allocation happens at **construction time** (B6: failure can only
      happen at that one startup moment, fail-fast); failure raises
      ``PinnedAllocationError``.
    - Freeing follows the tensors: once all views lose their references, the
      chunk returns to the host cache pool, and
      ``release_pinned_host_cache`` is what actually returns it to the OS
      (SS9.7 invariant).
    """

    #: Rounding granularity for the total: too coarse and a small
    #: configuration (anima's 8 layers, ~1GB) wastes a big chunk of locked
    #: memory; too fine and the chunk count and boundary fragmentation grow.
    #: 64MB stays within 0.3%-3% across every real configuration simulated.
    GRANULARITY = 64 * 1024 ** 2
    #: In-chunk offset alignment (any dtype's view is valid, and it's DMA-friendly)
    ALIGN = 256

    def __init__(
        self,
        total_bytes: int = 0,
        *,
        granularity: int = GRANULARITY,
        align: int = ALIGN,
        allocate=None,
    ) -> None:
        self._align = int(align)
        self._allocate = allocate or _alloc_pinned_bytes
        # [buffer(uint8 pinned), bytes used]
        self._chunks: list[list] = []
        self.packed_bytes = 0
        self.overflow_chunks = 0
        planned = 0
        if total_bytes > 0:
            planned = -(-int(total_bytes) // granularity) * granularity
        for size in _pow2_decomposition(planned):
            self._chunks.append([self._new_chunk(size), 0])

    def _new_chunk(self, nbytes: int) -> torch.Tensor:
        try:
            buf = self._allocate(nbytes)
        except RuntimeError as exc:  # cudaHostAlloc failed (torch.AcceleratorError is also a subclass of this)
            raise PinnedAllocationError(self.allocated_bytes + nbytes, str(exc)) from exc
        if buf.numel() != nbytes or buf.dtype != torch.uint8:
            raise RuntimeError("PinnedPacker's allocate must return nbytes worth of uint8")
        return buf

    @property
    def allocated_bytes(self) -> int:
        """Total bytes actually allocated (= actually locked)."""
        return sum(int(buf.numel()) for buf, _used in self._chunks)

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def _reserve(self, nbytes: int) -> tuple[torch.Tensor, int]:
        """Carve out ``nbytes`` from some chunk (best-fit + alignment), returning (chunk, start offset)."""
        best = None  # (remaining, chunk, start offset)
        for chunk in self._chunks:
            buf, used = chunk
            offset = -(-used // self._align) * self._align
            left = int(buf.numel()) - offset - nbytes
            if left >= 0 and (best is None or left < best[0]):
                best = (left, chunk, offset)
        if best is None:
            chunk = [self._new_chunk(_pow2_ceil(nbytes)), 0]
            self._chunks.append(chunk)
            self.overflow_chunks += 1
            offset = 0
        else:
            _left, chunk, offset = best
        chunk[1] = offset + nbytes
        self.packed_bytes += nbytes
        return chunk[0], offset

    def pin(self, tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Copy ``tensor``'s content into a pinned chunk, returning a
        same-shape pinned view.

        When ``dtype`` is given, casts along the way (the copy doubles as the
        conversion, saving an intermediate copy). ``tensor`` can be on any
        device (a GPU tensor is copied D2H straight into pinned memory).
        """
        target_dtype = dtype or tensor.dtype
        nbytes = tensor.numel() * torch.empty(0, dtype=target_dtype).element_size()
        buf, offset = self._reserve(nbytes)
        out = buf[offset: offset + nbytes].view(target_dtype).view(tuple(tensor.shape))
        out.copy_(tensor)
        return out


class PinnedBlockSwap:
    """Manages the swap-in/swap-out of weights for the trailing ``num_swap``
    blocks of an ``nn.ModuleList``.

    "Trailing" rather than an arbitrary subset: the DiT forward pass runs
    from 0 to N-1, so swapping out the later layers lets the earlier layers
    finish first, maximizing the time window freed up for the later ones
    (and matching musubi's ``blocks_to_swap`` semantics -- it also counts
    from the tail). The weights of the first ``N - num_swap`` blocks stay
    resident on GPU, unaffected.

    Lifecycle::
        swap = PinnedBlockSwap(blocks, num_swap, device)   # allocate pinned + GPU slots
        # each forward step:
        for i, block in enumerate(blocks):
            swap.ensure_resident(i)      # swapped-out layer -> ensure weights are on GPU (incl. waiting on prefetch)
            h = block(h, ...)
            swap.release(i)              # swapped-out layer -> mark its GPU slot reusable by the next layer
        # backward in reverse order works the same way (caller calls ensure_resident/release in reversed order)

    ensure_resident/release for a non-swapped layer (``i < first_swapped``)
    is a no-op, so callers can call them unconditionally without checking the
    boundary themselves.
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        num_swap: int,
        device: torch.device | str,
        *,
        num_slots: int = 2,
    ) -> None:
        total = len(blocks)
        if num_swap <= 0:
            raise ValueError("PinnedBlockSwap's num_swap must be positive (0 means this object shouldn't be constructed)")
        if num_swap > total:
            raise ValueError(
                f"num_swap={num_swap} exceeds the total block count {total}"
            )
        if num_slots < 2:
            raise ValueError("num_slots must be at least 2 (double buffering: compute current + prefetch next)")

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                f"block swap requires a CUDA device, got {self.device}"
            )
        self.blocks = blocks
        self.total = total
        self.num_swap = num_swap
        self.first_swapped = total - num_swap  # index of the first swapped-out block

        # Each swapped-out block's CPU pinned master weight copy:
        #   [block's relative index] -> {param name: pinned CPU tensor}
        # Uses a relative index (0 .. num_swap-1) to avoid confusion with the absolute index.
        self._cpu_weights: list[dict[str, torch.Tensor]] = []
        # Each param's shape/dtype metadata, used to build the matching buffer in a GPU slot
        self._param_specs: list[list[tuple[str, torch.Size, torch.dtype]]] = []

        # GPU slots: num_slots of them, each able to hold the full set of
        # params for any one swapped-out block.
        #   _slot_buffers[slot] = {param name: GPU tensor}
        self._slot_buffers: list[dict[str, torch.Tensor]] = []
        # Which relative block index each slot currently holds (-1 = empty)
        self._slot_holds: list[int] = [-1] * num_slots
        self.num_slots = num_slots

        self._copy_stream = torch.cuda.Stream(device=self.device)
        # ready[slot]: that slot's weight transfer is complete (the compute stream must wait on it before reading)
        self._ready = [torch.cuda.Event() for _ in range(num_slots)]
        # done[slot]: that slot's previous computation is complete (the copy stream must wait on it before overwriting, to prevent a data race)
        self._done = [torch.cuda.Event() for _ in range(num_slots)]

        self._pinned_bytes = 0
        self._handles: list = []
        self._build(blocks)

    # ------------------------------------------------------------------ construction
    def _build(self, blocks: nn.ModuleList) -> None:
        """Move the swapped-out blocks' weights to CPU pinned memory, and
        reserve slots on GPU.

        Pinned allocation failure is raised here (at startup, so it can fail-fast, doc SS8.1).
        """
        try:
            # All base weights not yet pinned are packed via PinnedPacker
            # (per-tensor pin_memory would get rounded up to a power of 2 by
            # the host allocator, wasting up to 1.47x, see PinnedPacker).
            # Count the total first, then allocate it in one shot: a failure
            # here fails fast, instead of blowing up after moving half the weights.
            to_pack = 0
            for absolute in range(self.first_swapped, self.total):
                for param in blocks[absolute].parameters():
                    if param.requires_grad or (
                        param.device.type == "cpu" and param.is_pinned()
                    ):
                        continue
                    to_pack += param.numel() * param.element_size()
            packer = PinnedPacker(to_pack) if to_pack > 0 else None

            for rel, absolute in enumerate(range(self.first_swapped, self.total)):
                block = blocks[absolute]
                cpu_w: dict[str, torch.Tensor] = {}
                specs: list[tuple[str, torch.Size, torch.dtype]] = []
                for name, param in block.named_parameters():
                    # **Only manages frozen base weights.** Trainable
                    # parameters (LoRA) must stay put and remain resident on
                    # GPU: they're the optimizer's target, and moving them
                    # would break training; also, relative to the base model
                    # they're tiny, so there's no benefit in swapping them out.
                    if param.requires_grad:
                        continue
                    # If it's already on CPU and already pinned, take it over
                    # in place (the loader can load the trailing layers
                    # straight into CPU pinned memory, so GPU peak usage
                    # never passes through the full model -- the precondition
                    # for the 12/16GB targets); otherwise (pageable CPU
                    # memory, or still on GPU) pack it into a pinned chunk.
                    src = param.detach()
                    if src.device.type == "cpu" and src.is_pinned():
                        pinned = src
                    else:
                        pinned = packer.pin(src)
                    cpu_w[name] = pinned
                    specs.append((name, param.shape, param.dtype))
                    self._pinned_bytes += pinned.numel() * pinned.element_size()
                    # The master copy is now in CPU pinned memory -- release
                    # the original GPU weight immediately (that's where the
                    # VRAM savings come from; otherwise it wouldn't be
                    # realized until the first forward rebind). .data points
                    # at this pinned CPU tensor (rather than empty(0)):
                    # keeping shape/dtype lets LyCORIS, which is injected
                    # **after construction**, correctly read the base
                    # weight's shape; a later ensure_resident then switches
                    # .data to the GPU slot.
                    param.data = pinned
                self._cpu_weights.append(cpu_w)
                self._param_specs.append(specs)
            torch.cuda.empty_cache()  # return the just-freed original weight segments to the allocator

            # Reserve GPU slots: capacity = the largest of the swapped-out
            # blocks (they're all the same size in a homogeneous DiT)
            for _slot in range(self.num_slots):
                buf: dict[str, torch.Tensor] = {}
                for name, shape, dtype in self._param_specs[0]:
                    buf[name] = torch.empty(shape, dtype=dtype, device=self.device)
                self._slot_buffers.append(buf)
        except RuntimeError as exc:  # pinned / GPU allocation failed
            self._pinned_bytes = 0
            raise BlockSwapAllocationError(
                self.num_swap, self.first_swapped, str(exc)
            ) from exc

        logger.info(
            "block swap ready: swapping out the trailing %d/%d blocks, pinned %.2f GB, %d GPU slots",
            self.num_swap, self.total, self._pinned_bytes / _GIB, self.num_slots,
        )

    @property
    def pinned_bytes(self) -> int:
        """Total bytes of the CPU pinned master copies (basis for the guardrail budget)."""
        return self._pinned_bytes

    # ------------------------------------------------------------------ primitives
    def _slot_for(self, rel: int) -> int:
        """Relative index -> the GPU slot to use (rel's parity under double buffering)."""
        return rel % self.num_slots

    def _fetch(self, rel: int) -> None:
        """On the copy stream, move the rel-th swapped-out block's weights into its slot (if not already resident)."""
        if rel < 0 or rel >= self.num_swap:
            return
        slot = self._slot_for(rel)
        if self._slot_holds[slot] == rel:
            return  # already resident (usually a prefetch hit from the previous step)
        with torch.cuda.stream(self._copy_stream):
            # That slot's previous computation must finish first, or it would
            # overwrite weights currently being read (a data race). An
            # Event.wait on an event that was never recorded is a no-op, so
            # the first round is naturally safe.
            self._copy_stream.wait_event(self._done[slot])
            buf = self._slot_buffers[slot]
            src = self._cpu_weights[rel]
            for name, dst in buf.items():
                dst.copy_(src[name], non_blocking=True)
            self._ready[slot].record(self._copy_stream)
        self._slot_holds[slot] = rel
        self._rebind(rel, slot)

    def _rebind(self, rel: int, slot: int) -> None:
        """Point that block's base weights' ``.data`` at the slot buffer (an
        in-place swap, not a module swap).

        Only rebinds parameter names **registered at construction time**:
        parameters added after construction (the case where LoRA injects new
        submodules inside the block) are not managed by this component, and
        iterating named_parameters() would run into them.
        """
        block = self.blocks[self.first_swapped + rel]
        buf = self._slot_buffers[slot]
        params = dict(block.named_parameters())
        for name, _shape, _dtype in self._param_specs[rel]:
            param = params.get(name)
            if param is not None:
                param.data = buf[name]

    def ensure_resident(self, absolute_index: int, *, prefetch_next: int | None = None) -> None:
        """Ensure the ``absolute_index``-th block's weights are on GPU and
        safe for the compute stream to read.

        A no-op for non-swapped-out (resident) layers. If ``prefetch_next``
        is given (the absolute index of the next block to be used), also
        kicks off its prefetch -- this is key to hiding the transfer; the
        caller should pass i+1 for forward or i-1 for backward.
        """
        rel = absolute_index - self.first_swapped
        if rel < 0:
            return  # resident layer
        self._fetch(rel)
        if prefetch_next is not None:
            nxt = prefetch_next - self.first_swapped
            if 0 <= nxt < self.num_swap:
                self._fetch(nxt)
        torch.cuda.current_stream().wait_event(self._ready[self._slot_for(rel)])

    def release(self, absolute_index: int) -> None:
        """Mark that this block's computation has been issued on the compute
        stream, so its GPU slot can be overwritten by a later block.

        A no-op for non-swapped-out layers. Must be called after that
        block's forward call.
        """
        rel = absolute_index - self.first_swapped
        if rel < 0:
            return
        self._done[self._slot_for(rel)].record(torch.cuda.current_stream())

    def reset(self) -> None:
        """Reset slot-occupancy state before one step (forward or backward) begins.

        The double-buffered slots hold the last two layers from the end of
        the previous step; a new step starts from the head/tail and needs to
        prefetch again. Doesn't free any VRAM, just clears the hold markers.
        """
        self._slot_holds = [-1] * self.num_slots

    def restore_masters(self) -> None:
        """Point every managed parameter's ``.data`` back at the CPU pinned master copy.

        Required on the inference side (doc SS9.6). Two scenarios:

        1. **fp8 LoRA merge**: merging writes to ``module.weight``. If
           ``.data`` currently points at a GPU slot, the delta just written
           gets **directly overwritten** by the next layer's swap-in --
           the merge is silently lost. Restore first, then merge, so the
           delta lands on the master copy, and every subsequent swap-in
           brings in the already-merged weight.
        2. **Any external operation that reads/modifies weights** (export,
           inspection, requantization): the master copy is the only complete
           and stable one; the GPU slots are just a rotating window.
        """
        for rel in range(self.num_swap):
            block = self.blocks[self.first_swapped + rel]
            params = dict(block.named_parameters())
            master = self._cpu_weights[rel]
            for name, _shape, _dtype in self._param_specs[rel]:
                param = params.get(name)
                if param is not None:
                    param.data = master[name]
        # Slot contents are now unbound from any param, mark them empty to avoid a false hit next time
        self._slot_holds = [-1] * self.num_slots

    def managed_data_ptrs(self) -> set[int]:
        """The set of **storage addresses** for tensors managed by this
        component (CPU master copies + GPU slots).

        For external "move the whole model" operations to use: these tensors
        **must not** be moved onto GPU by a blanket ``module.to(device)`` or
        similar, or block swap is wasted (see ``move_module_excluding``).

        Uses ``data_ptr()`` rather than ``id()``: every access of
        ``param.data`` returns a **new** Python wrapper object, so ``id()``
        is unstable and comparing against it would miss every match.
        """
        ptrs = set()
        for weights in self._cpu_weights:
            ptrs.update(t.data_ptr() for t in weights.values())
        for buf in self._slot_buffers:
            ptrs.update(t.data_ptr() for t in buf.values())
        return ptrs

    def attach(self) -> None:
        """Register forward + backward hooks on every swapped-out block, taking over swap-in/swap-out.

        This is the **recommended wiring approach**: it takes effect entirely
        from the outside, without needing to modify the model's forward loop
        (krea2's loop lives inside the parity-sensitive ``modeling/``, which
        shouldn't be touched, doc SS7.1).

        **All four hooks are required** (forward pre/post + backward pre/post):

        - forward pre fetches the weights back, post releases the slot;
        - **backward pre must fetch again** -- forward post has already
          released the slot (without releasing it, there would be no done
          event at all during the forward pass, and double buffering would
          lose its protection), and by the time this block's backward runs,
          the slot already holds some other layer.

        It was once assumed that "turning on gradient checkpointing means the
        backward recompute triggers the forward hook, so the reverse-order
        swap-in happens automatically" -- **that's wrong**: the recomputed
        forward_hook doesn't fire at all (measured: under checkpointing, pre
        fires 2N times but post only N times), and even after the recompute,
        this block's backward still needs to read the weights. Without the
        backward hook, gradients get computed **silently** wrong -- no error,
        no NaN, just incorrect numbers; on real hardware the measured
        deviation reached 300x the noise floor, immediately blowing up PPSF's
        d estimate. See the regression test
        ``tests/test_block_swap_grad_fidelity.py`` (must use real-world
        sizes + noise-floor calibration; small-tensor tests are completely
        insensitive to this).

        Idempotent: calling it repeatedly does not re-register.
        """
        if self._handles:
            return
        for rel in range(self.num_swap):
            absolute = self.first_swapped + rel
            block = self.blocks[absolute]

            def pre_hook(_module, _args, idx=absolute):
                self.ensure_resident(idx, prefetch_next=idx + 1)

            def post_hook(_module, _args, output, idx=absolute):
                self.release(idx)
                return output

            def backward_pre_hook(_module, _gout, idx=absolute):
                # **Backward must fetch the weights back itself.** By the
                # time the forward pass ends the slot is already released,
                # and by the time this block's backward runs, the slot
                # already holds some other layer -- skipping this step
                # would make gradients **silently** wrong (no error, no NaN,
                # just incorrect numbers; on real hardware the measured
                # deviation reached 300x the noise floor, immediately
                # blowing up PPSF's d estimate). With checkpointing on, the
                # immediately preceding recompute has just put the weights
                # back in place, so this call hits directly with zero
                # extra transfer.
                self.ensure_resident(idx, prefetch_next=idx - 1)

            def backward_hook(_module, _gin, _gout, idx=absolute):
                self.release(idx)

            self._handles.append(block.register_forward_pre_hook(pre_hook))
            self._handles.append(block.register_forward_hook(post_hook))
            self._handles.append(block.register_full_backward_pre_hook(backward_pre_hook))
            self._handles.append(block.register_full_backward_hook(backward_hook))
        logger.info("block swap attached: forward/backward hooks on %d blocks", self.num_swap)

    def detach(self) -> None:
        """Remove the hooks registered by attach (weights are not restored; call ensure_resident yourself if needed)."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def close(self) -> None:
        """Let go entirely: remove hooks + point managed parameters at empty
        tensors + discard the master copies and GPU slots.

        (Deliberately not named ``release`` -- that name is already used for
        the per-layer primitive "a layer is done computing, its slot can be
        reused", a completely different semantics.)

        **The model is unusable after this call**; only call it when you're
        certain you're done with it (end of training / model unload).

        Why this is needed, rather than just "drop the reference to model":
        the pinned master copies are referenced by param.data, and
        `ctx.model` is not the only thing holding the block -- the LyCORIS
        injector holds org_module, the optimizer holds the parameters, and
        hook closures may hold references too. On real hardware, dropping
        just the swap object returns 0 bytes; only dropping the model
        afterward returns anything. Rather than hunting down every holder,
        this object proactively redirects its parameters away.

        Note ``release_pinned_host_cache()`` still needs to be called
        afterward to actually return the memory to the OS (that's another
        layer, the host caching allocator, doc SS9.7).
        """
        self.detach()
        for rel in range(min(self.num_swap, len(self._param_specs))):  # idempotent
            block = self.blocks[self.first_swapped + rel]
            params = dict(block.named_parameters())
            for name, _shape, dtype in self._param_specs[rel]:
                param = params.get(name)
                if param is not None:
                    param.data = torch.empty(0, dtype=dtype, device=self.device)
        self._cpu_weights.clear()
        self._slot_buffers.clear()
        self._param_specs.clear()
        self._slot_holds = []
        self._pinned_bytes = 0

    def iter_forward(self) -> Iterator[tuple[int, nn.Module]]:
        """Convenience wrapper for the forward traversal: yields (index,
        block), automatically doing ensure_resident+prefetch+release.

        Caller: ``for i, block in swap.iter_forward(): h = block(h, ...)``.
        Note release is called after the yield returns, so the caller must
        finish the forward pass within the loop body.
        """
        self.reset()
        for i in range(self.total):
            self.ensure_resident(i, prefetch_next=i + 1)
            yield i, self.blocks[i]
            self.release(i)


def release_pinned_host_cache() -> None:
    """Return pinned (page-locked) memory to the operating system.

    This is a **different thing** from ``torch.cuda.empty_cache()``: the
    latter only manages the device side. Pinned memory goes through
    PyTorch's separate host caching allocator, and freeing a tensor just
    returns it to that cache pool -- on real hardware, pinning 6GB and then
    ``del`` + ``gc.collect()`` returns **0 bytes**; only calling this
    function returns the 8GB (doc SS9.7).

    Block swap's master copies can reach 11GB+; missing this step means
    "unloaded but memory not returned", and page-locked memory can't even be
    paged out, so other programs can't use it at all. This is the host-side
    version of the same class of problem as
    ``_cuda_clearCublasWorkspaces`` (state resident at the C++/allocator
    layer, invisible to Python's GC).

    **When to call this**: only once you're certain that batch of weights is
    no longer needed (model unload / end of training). Never call it during
    image generation or training -- what's in pinned memory is the model
    weights themselves.

    Internal API; silently skipped if missing/failing (the next load will
    just reuse the cache, only the memory isn't returned to the system).
    """
    try:
        torch._C._host_emptyCache()
    except Exception:  # noqa: BLE001
        pass


def move_module_excluding(module: nn.Module, device, swap: "PinnedBlockSwap | None") -> None:
    """Move ``module`` to ``device``, but **skip the parameters managed by block swap**.

    On the inference side, the daemon moves the whole model back to GPU
    before each task (after sampling-time offload, it needs to come back).
    That's a blanket ``module.to(device)`` -- under block swap that would
    also drag the swapped-out layers' CPU pinned master copies onto the
    card, wasting the swap entirely, and the instantaneous memory usage
    would equal the full model, causing an immediate OOM on a 12GB card.

    When ``swap`` is None, this degrades to a plain ``module.to(device)``
    (zero behavior change).
    """
    if module is None or not hasattr(module, "to"):
        return
    if swap is None:
        module.to(device)
        return
    managed = swap.managed_data_ptrs()
    target = torch.device(device)
    for _name, param in module.named_parameters(recurse=True):
        if param.data.data_ptr() in managed or param.data.device == target:
            continue
        param.data = param.data.to(target)
        if param.grad is not None:
            param.grad = param.grad.to(target)
    for _name, buf in module.named_buffers(recurse=True):
        if buf.data_ptr() in managed or buf.device == target:
            continue
        # A buffer needs to be re-registered through its owning module to
        # swap the instance; directly modifying .data also works for a plain
        # (non-Parameter) Tensor (which is what a buffer stores)
        buf.data = buf.data.to(target)


class BlockSwapAllocationError(RuntimeError):
    """pinned / GPU slot allocation failed (doc SS8.1 / B6: error out, never silently degrade).

    Can only be raised during ``PinnedBlockSwap`` construction at startup.
    Carries enough context for the caller to surface an actionable message
    to the user (close memory-hungry applications / lower blocks_to_swap).
    """

    def __init__(self, num_swap: int, first_swapped: int, detail: str) -> None:
        self.num_swap = num_swap
        self.first_swapped = first_swapped
        self.detail = detail
        super().__init__(
            f"block swap pre-allocation failed (swapping out the trailing {num_swap} blocks): {detail}"
        )

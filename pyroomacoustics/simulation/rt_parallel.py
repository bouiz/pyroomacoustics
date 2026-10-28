"""
Multi-process ray tracing.

The rays of one source are independent and the histograms they fill are
additive, so the rays are split across worker processes, each of which traces its
slice in its own copy of the C++ room, and the histograms are summed.
Per-ray energies are scaled by the slice size (as the chunked serial loop in
:py:meth:`pyroomacoustics.room.Room.ray_tracing` does), hence the sum equals
what a single process would have collected, up to Monte-Carlo noise.

Workers are spawned (never forked) and cached per process count. They are
started from a thread-safe, lazily created ``ProcessPoolExecutor`` and watch
their parent: if it is killed (a cancelled job, for instance) they exit on
their own instead of finishing a ray-tracing batch nobody waits for.
"""

import contextlib
import multiprocessing
import multiprocessing.util
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Below this many rays per process the spawn + room rebuild costs more than it saves.
MIN_RAYS_PER_PROCESS = 500

# Tasks per process: a few more than one so progress is reported at a finer grain
# and a slow slice does not leave the other processes idle.
SLICES_PER_PROCESS = 2

_pools = {}
_pools_lock = threading.Lock()


def _parent_alive_watch(parent_pid):
    """Hard-exit this process once ``parent_pid`` is gone."""
    if sys.platform == "win32":
        import ctypes

        SYNCHRONIZE = 0x00100000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid)
        if not handle:  # already gone
            os._exit(1)
        kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        os._exit(1)
    while os.getppid() == parent_pid:
        time.sleep(1.0)
    os._exit(1)


def _init_worker(parent_pid):
    threading.Thread(target=_parent_alive_watch, args=(parent_pid,), daemon=True).start()


@contextlib.contextmanager
def _children_allowed():
    """
    Context in which the pool's workers may be started.

    * ``multiprocessing`` refuses to start children from a daemonic process, and
      job runners commonly run simulations in one. The pool's workers watch their
      parent themselves, so the restriction (meant to prevent orphans) does not apply.
    * Spawned workers re-import the parent's ``__main__`` module, which for a job
      runner means its whole application (seconds per worker). They only need this
      package, so ``__main__`` is hidden from the spawn machinery while they start.
    """
    config = getattr(multiprocessing.current_process(), "_config", None)
    was_daemon = bool(config and config.get("daemon"))
    if was_daemon:
        config["daemon"] = False

    main = sys.modules.get("__main__")
    hidden = {name: getattr(main, name) for name in ("__spec__", "__file__") if hasattr(main, name)}
    for name in hidden:
        if name == "__spec__":
            main.__spec__ = None
        else:
            del main.__file__
    try:
        yield
    finally:
        for name, value in hidden.items():
            setattr(main, name, value)
        if was_daemon:
            config["daemon"] = True


def get_pool(n_processes):
    with _pools_lock:
        pool = _pools.get(n_processes)
        if pool is None:
            pool = ProcessPoolExecutor(
                max_workers=n_processes,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
                initargs=(os.getpid(),),
            )
            _pools[n_processes] = pool
        return pool


def shutdown_pools():
    with _pools_lock:
        pools = list(_pools.values())
        _pools.clear()
    for pool in pools:
        pool.shutdown(wait=True, cancel_futures=True)


# The pool's workers are non-daemonic children. A multiprocessing child joins all
# of them in its exit hook *before* ``atexit`` handlers run, so with a plain
# ``atexit`` shutdown it would wait forever on idle workers. Finalizers with a
# non-negative exit priority run before that join.
multiprocessing.util.Finalize(None, shutdown_pools, exitpriority=100)


def build_engine_spec(room):
    """
    Picklable description of the room's C++ engine, or None when the room cannot
    be rebuilt in a worker (2D and shoebox rooms: their ray tracing is cheap).
    """
    if room.dim != 3 or getattr(room, "_engine_args", None) is None or room.walls is None:
        return None
    if getattr(room, "_engine_is_polygon", False) is not True:
        return None
    return {
        "walls": [
            (
                np.array(w.corners, dtype=np.float32),
                np.array(w.absorption, dtype=np.float32),
                np.array(w.scatter, dtype=np.float32),
                str(w.name),
                bool(w.two_sided),
            )
            for w in room.walls
        ],
        "obstructing": [int(i) for i in room.room_engine.obstructing_walls],
        "params": (
            float(room.c),
            int(room.max_order),
            float(room.rt_args["energy_thres"]),
            float(room.rt_args["time_thres"]),
            float(room.rt_args["receiver_radius"]),
            float(room.rt_args["hist_bin_size"]),
            bool(room.simulator_state["ism_needed"] and room.simulator_state["rt_needed"]),
        ),
    }


def _trace_slice(job):
    """Worker: rebuild the room, trace one slice of rays, return every mic's histograms."""
    from .. import libroom
    from ..room import wall_factory

    spec, mics, dirs, energies, position, seed = job
    libroom.set_rng_seed(int(seed))

    walls = []
    for corners, absorption, scatter, name, two_sided in spec["walls"]:
        wall = wall_factory(corners, absorption, scatter, name)
        if two_sided:
            wall.two_sided = True
        walls.append(wall)
    engine = libroom.Room(walls, spec["obstructing"], [], *spec["params"])

    # All microphones are added before any is configured: the C++ engine stores
    # them in a vector, and configuring while it still grows crashes the tracer.
    for loc, _ in mics:
        engine.add_mic(np.asarray(loc, dtype=np.float32).reshape(3, 1))
    for k, (_, directions) in enumerate(mics):
        if directions is None:
            engine.microphones[k].make_omnidirectional()
        else:
            engine.microphones[k].set_directions(directions)

    engine.ray_tracing(dirs, energies, np.asarray(position, dtype=np.float32))
    return [[_to_sparse(h.get_hist()) for h in mic.histograms] for mic in engine.microphones]


def _to_sparse(hist):
    """
    (shape, flat indices, values) of the non-zero cells. A directional receiver has
    hundreds of (bands, bins) histograms that are mostly empty, so shipping them
    dense back to the parent would cost more than the tracing saves.
    """
    hist = np.asarray(hist, dtype=np.float32)
    flat = hist.reshape(-1)
    idx = np.flatnonzero(flat)
    return hist.shape, idx.astype(np.int64), flat[idx]


def _accumulate(acc, part):
    """Sum a worker's sparse histograms into ``acc`` (per mic, per direction dense arrays or None)."""
    for m, mic_hists in enumerate(part):
        if acc[m] is None:
            acc[m] = [None] * len(mic_hists)
        for d, (shape, idx, values) in enumerate(mic_hists):
            dense = acc[m][d]
            if dense is None:
                dense = np.zeros(shape, dtype=np.float32)
            elif dense.shape[-1] < shape[-1]:  # a longer histogram: the rays that travelled farther
                dense = np.pad(dense, [(0, 0)] * (dense.ndim - 1) + [(0, shape[-1] - dense.shape[-1])])
            if idx.size:
                dense[np.unravel_index(idx, shape)] += values  # indices are unique within a slice
            acc[m][d] = dense


def trace_parallel(spec, mics, dirs, energies, position, n_processes, seeds, progress=None):
    """
    Trace ``dirs`` (n_rays, 3) from ``position`` over ``n_processes`` processes.

    Parameters
    ----------
    spec: dict
        From :py:func:`build_engine_spec`.
    mics: list of (location, directions)
        One entry per engine microphone; ``directions`` is None for an
        omnidirectional receiver.
    energies: (n_rays, n_bands) array
        Relative energy of each ray, as passed to the C++ engine on the serial path.
    seeds: sequence of int
        One RNG seed per slice (``SLICES_PER_PROCESS * n_processes`` of them are used).
    progress: callable, optional
        Called as ``progress(done_rays, total_rays)`` when a slice completes.

    Returns
    -------
    list (per mic) of lists (per direction) of (bands, bins) arrays
    """
    n_total = dirs.shape[0]
    n_slices = max(1, min(n_processes * SLICES_PER_PROCESS, n_total // MIN_RAYS_PER_PROCESS))
    slices = [np.arange(i, n_total, n_slices) for i in range(n_slices)]

    jobs = []
    for i, idx in enumerate(slices):
        # The C++ engine divides the initial energy by the number of rays it is
        # given; this keeps the per-ray energy at ENERGY_0 / n_total.
        jobs.append(
            (
                spec,
                mics,
                np.ascontiguousarray(dirs[idx], dtype=np.float32),
                np.ascontiguousarray(energies[idx] * (len(idx) / n_total), dtype=np.float32),
                np.asarray(position, dtype=np.float32),
                int(seeds[i]),
            )
        )

    pool = get_pool(n_processes)
    with _children_allowed():
        futures = {pool.submit(_trace_slice, job): len(slices[i]) for i, job in enumerate(jobs)}

    acc = [None] * len(mics)
    done_rays = 0
    for future in as_completed(futures):
        _accumulate(acc, future.result())
        done_rays += futures[future]
        if progress is not None:
            progress(done_rays, n_total)
    return acc

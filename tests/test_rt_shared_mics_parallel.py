"""Ray tracing with shared coincident receivers and several processes, ISM with a shadowed microphone."""

import types
import warnings

import numpy as np
import pytest

import pyroomacoustics as pra
from pyroomacoustics.directivities import CardioidFamily, DirectionVector
from pyroomacoustics.simulation import compute_ism_rir

FS = 16000


def _room(n_rays=4000, **ray_kwargs):
    material = pra.Material(energy_absorption=0.25, scattering=0.2)
    outline = np.array([[0, 0], [8, 0], [8, 3], [3, 3], [3, 8], [0, 8]], dtype=float).T
    room = pra.Room.from_corners(
        outline, materials=material, fs=FS, max_order=2, ray_tracing=True, air_absorption=False
    )
    room.extrude(3.0, materials=material)
    room.set_ray_tracing(n_rays=n_rays, receiver_radius=0.5, **ray_kwargs)
    room.add_source([1.0, 1.0, 1.5])
    return room


def _foa_array(position):
    directions = [
        CardioidFamily(DirectionVector(azimuth=a, colatitude=c, degrees=True), p=p)
        for a, c, p in [(0, 0, 1.0), (0, 90, 0.0), (0, 0, 0.0), (-90, 90, 0.0)]
    ]
    return np.column_stack([position] * 4), directions


def _add_mics(room):
    positions, directivities = _foa_array([5.0, 1.5, 1.5])
    room.add_microphone_array(positions, directivity=directivities)
    room.add_microphone([2.0, 5.0, 1.5])  # one omnidirectional mic elsewhere


def _traced_histograms(**ray_kwargs):
    pra.random.seed(numpy=3, libroom=3)
    room = _room(**ray_kwargs)
    _add_mics(room)
    room.image_source_model()
    room.ray_tracing()
    return room


def test_coincident_directional_mics_share_one_receiver():
    shared = _traced_histograms(receiver_n_directions=30)
    separate = _traced_histograms(receiver_n_directions=30, share_coincident_mics=False)

    # The four channels collect the same histograms...
    for m in range(1, 4):
        assert shared.rt_histograms[m][0][0] is shared.rt_histograms[0][0][0]
    # ...which are exactly what each of them gets when traced separately.
    for m in range(5):
        for h_shared, h_separate in zip(shared.rt_histograms[m][0], separate.rt_histograms[m][0]):
            np.testing.assert_array_equal(h_shared, h_separate)
    # The channels keep their own directional response.
    responses = [r["response"] for r in shared.rt_directional_responses[:4]]
    assert not np.allclose(responses[0], responses[1])


def test_shared_receiver_engine_does_not_touch_the_room_engine():
    room = _traced_histograms(receiver_n_directions=30)
    assert len(room.room_engine.microphones) == 5
    # still usable: a full RIR can be built from the shared histograms
    room.compute_rir()
    assert all(len(room.rir[m][0]) > 0 for m in range(5))


@pytest.mark.parametrize("n_processes", [2])
def test_multi_process_ray_tracing_matches_single_process(n_processes):
    single = _traced_histograms(n_rays=6000, receiver_n_directions=30)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a failed pool falls back to one process with a warning
        multi = _traced_histograms(
            n_rays=6000, receiver_n_directions=30, n_processes=n_processes
        )

    for m in (0, 4):  # a directional and an omnidirectional receiver
        a = sum(h.sum() for h in single.rt_histograms[m][0])
        b = sum(h.sum() for h in multi.rt_histograms[m][0])
        assert a > 0 and b > 0
        assert 0.6 < b / a < 1.6
    assert len(multi.rt_histograms[0][0]) == len(single.rt_histograms[0][0])


def test_ism_microphone_without_visible_image_returns_none():
    src = types.SimpleNamespace(images=np.zeros((3, 2)), damping=np.ones((1, 2)), directivity=None)
    out = compute_ism_rir(
        src, np.zeros(3), None, np.zeros((3, 2)), np.zeros(2, dtype=bool), 81, 343.0, FS, None
    )
    assert out is None


def test_room_survives_a_microphone_that_nothing_reaches():
    room = _room()
    room.add_microphone([5.0, 1.5, 1.5])
    room.image_source_model()
    room.visibility[0][:] = 0  # nothing visible from the microphone
    room.rt_args["n_processes"] = 1
    room.ray_tracing()
    room.compute_rir()
    assert len(room.rir[0][0]) > 0  # the ray-traced tail alone

"""``Room.check_inside``: off by default (open architectural meshes), upstream behaviour when on."""

import numpy as np
import pytest

import pyroomacoustics as pra


def _room(check_inside):
    room = pra.Room.from_corners(np.array([[0, 0, 6, 6], [0, 6, 6, 0]]), max_order=2)
    room.extrude(6)
    room.check_inside = check_inside
    return room


def test_default_keeps_points_outside_the_room():
    room = _room(check_inside=False)
    room.add_source([7.0, 7.0, 7.0])
    room.add_microphone([3.0, 3.0, 3.0])
    room.add_microphone([8.0, 8.0, 8.0])
    room.image_source_model()
    assert room.visibility[0][1].any()


def test_check_inside_rejects_and_silences_points_outside():
    room = _room(check_inside=True)
    with pytest.raises(ValueError):
        room.add_source([7.0, 7.0, 7.0])
    room.add_source([3.0, 3.0, 3.0])
    room.add_microphone([2.0, 2.0, 2.0])
    room.add_microphone([8.0, 8.0, 8.0])
    room.image_source_model()
    assert room.visibility[0][0].any()
    assert not room.visibility[0][1].any()

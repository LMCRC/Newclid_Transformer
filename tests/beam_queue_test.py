from alphageo.alphageometry import BeamQueue


def test_beam_queue():
    beam_queue = BeamQueue(max_size=2)

    beam_queue.add(1, "a")
    beam_queue.add(2, "b")
    beam_queue.add(3, "c")

    beam_queue = list(beam_queue)
    assert beam_queue == [(3, "c"), (2, "b")]

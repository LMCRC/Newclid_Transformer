from alphageo.alphageometry import BeamQueue


def test_beam_queue():
    beam_queue = BeamQueue(max_size=2)

    beam_queue.add("a", 1)
    beam_queue.add("b", 2)
    beam_queue.add("c", 3)

    beam_queue = list(beam_queue)
    assert beam_queue == [(3, "c"), (2, "b")]

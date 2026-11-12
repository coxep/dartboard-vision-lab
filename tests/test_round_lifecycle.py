def test_round_lifecycle():
    from dartboard_vision.session import RoundState

    def contact(x, y):
        return dict(point_pixels=[x, y], support=2)

    s = RoundState(score_point=lambda p: dict(score=20))
    for t in range(3):
        assert s.observe([], [0.0, 0.0, 0.0], t)["round_total"] == 0
    a = contact(400, 250)
    b = contact(600, 350)
    assert not s.observe([a], [0.01, 0.01, 0.01], 3)["hits"]
    assert len(s.observe([a], [0.01, 0.01, 0.01], 4)["hits"]) == 1
    assert s.observe([a, b], [0.02, 0.02, 0.02], 5)["round_total"] == 20
    assert s.observe([a, b], [0.02, 0.02, 0.02], 6)["round_total"] == 40
    assert len(s.observe([b], [0.02, 0.02, 0.02], 7)["hits"]) == 2
    # Zero proposed contacts while a hand/darts still change the image is not clear.
    for t in range(8, 12):
        assert len(s.observe([], [0.05, 0.02, 0.03], t)["hits"]) == 2
    # An invalid alignment must not clear the board either.
    for t in range(12, 16):
        assert len(s.observe([], [0.0, 0.0, 0.0], t, False)["hits"]) == 2
    for t in (16, 17):
        assert s.observe([], [0.0, 0.0, 0.0], t)["round_total"] == 40
    r = s.observe([], [0.0, 0.0, 0.0], 18)
    assert r["round_total"] == 0 and not r["hits"] and r["round_id"] == 2
    assert len(s.history) == 1 and s.history[0]["round_total"] == 40
    assert r["events"][0]["type"] == "round_reset"
    u = RoundState()
    u.observe([a], [0.01, 0.01], 0)
    r = u.observe([a], [0.01, 0.01], 1)
    assert r["round_total"] is None and not r["score_verified"]
    print(
        "New hits, one-to-one association, missing/occluded views, alignment hold and clear-board score reset passed."
    )

from fuzzywuzzy import fuzz

RESEMBLE_MIN_DEGREE = 63
VERY_RESEMBLE_DEGREE = 85


def compare_ratio(name1, name2):
    return resemble_mix_ratio(name1, name2)


def resemble_mix_ratio(name1, name2):
    name1st = name1.lower().replace("-", " ")
    name2nd = name2.lower().replace("-", " ")
    return int(
        round(
            (fuzz.ratio(name1st, name2nd) + fuzz.partial_ratio(name1st, name2nd)) * 0.5
        )
    )


def resemble_ratio(name1, name2):
    name1st = name1.lower().replace("-", " ")
    name2nd = name2.lower().replace("-", " ")
    return fuzz.ratio(name1st, name2nd)


def resemble_partial_ratio(name1, name2):
    name1st = name1.lower().replace("-", " ")
    name2nd = name2.lower().replace("-", " ")
    return fuzz.partial_ratio(name1st, name2nd)


def test_simple0():
    n1, n2 = "Jaume Munar", "Jaume Antoni Munar Clar"
    r = compare_ratio(n1, n2)
    print(r, f"'{n1}' | '{n2}'")
    assert r >= 60


def test_simple():
    n1, n2 = "Rogerio Dutra Silva", "Daniel Dutra da Silva"  # real atp
    r = compare_ratio(n1, n2)
    print(r, f"'{n1}' | '{n2}'")
    assert r < 70


def test_simple2():
    n1, n2 = "Rogerio Dutra Silva", "Rogerio Dutra da Silva"
    r = compare_ratio(n1, n2)
    print(r, f"'{n1}' | '{n2}'")
    assert r >= 88


def test_simple3():
    n1, n2 = "Arklon Huertas Del Pino", "Arklon Huertas Del Pino Cordova"  # real atp
    r = compare_ratio(n1, n2)
    print(r, f"'{n1}' | '{n2}'")
    assert r >= 92


def test_simple4():
    n1, n2 = "Alexander Lazov", "Alexandar Lazov"  # real atp
    r = compare_ratio(n1, n2)
    print(r, f"'{n1}' | '{n2}'")
    assert r >= 93


def test_simple5():
    n1, n2 = "Moez Chargui", "Moez Echargui"  # real atp
    r = compare_ratio(n1, n2)
    print(r, f"'{n1}' | '{n2}'")
    assert r >= 94


def test_simple6():
    n1, n2 = "N. Vijay Sundar Prashanth", "N Vijay Sundar Prashanth"  # real atp
    r = compare_ratio(n1, n2)
    print(r, f"'{n1}' | '{n2}'")
    assert r >= 96


def test_simple7():
    n1, n2 = "Bruno Sant'Anna", "Bruno Santanna"  # real atp
    r = compare_ratio(n1, n2)
    print(r, f'"{n1}" | "{n2}"')
    assert r >= 94


def test_simple8():
    n1, n2 = "Samuel Groth", "Sam Groth"  # real atp
    r = compare_ratio(n1, n2)
    print(r, f'"{n1}" | "{n2}"')
    assert r >= 76


def test_simple9():
    n1, n2 = "Aleksandr Nedovesov", "Aleksandr Nedovyesov"  # real atp
    r = compare_ratio(n1, n2)
    print(r, f'"{n1}" | "{n2}"')
    assert r >= 95


def test_simple10():
    n1, n2 = "Dmitry Trunov", "Dmitry Tursunov"  # real atp
    r = compare_ratio(n1, n2)
    print(r, f'"{n1}" | "{n2}"')
    assert r <= 89


def test_simple11():
    n1, n2 = "Richard Berankis", "Ricardas Berankis"  # real atp
    r = compare_ratio(n1, n2)
    print(r, f'"{n1}" | "{n2}"')
    assert r >= 90


def test_simple12():
    n1, n2 = "Mathieu Roy", "Matthieu Roy"  # real atp
    r = compare_ratio(n1, n2)
    print(r, f'"{n1}" | "{n2}"')
    assert r >= 94

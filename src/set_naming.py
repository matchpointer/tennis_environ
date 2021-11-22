OPEN = 1
OPEN2 = 2  # best_of_five only third set after 1:1
DECIDED = (3,)
PRESS = 4
PRESS2 = 5  # best_of_five only third set after 2:0
UNDER = 6
UNDER2 = 7  # best_of_five only third set after 0:2

name_to_code = {
    "open": OPEN,
    "open2": OPEN2,
    "decided": DECIDED,
    "press": PRESS,
    "press2": PRESS2,
    "under": UNDER,
    "under2": UNDER2,
}


def get_code(name):
    if isinstance(name, tuple):
        return name_to_code.get(name[0], None)
    return name_to_code.get(name, None)


def get_opponent_code(code):
    if code in (OPEN, OPEN2, DECIDED):
        return code
    elif code == PRESS:
        return UNDER
    elif code == PRESS2:
        return UNDER2
    elif code == UNDER:
        return PRESS
    elif code == UNDER2:
        return PRESS2


def get_names(scr, setnum, best_of_five=None):
    """:returns 2-tuple of text_set_names for left and for right player
    :arg scr is score.Score
    :arg setnum is number of set (1-based)
    """
    cursetnum, win_sets_count, loss_sets_count = 0, 0, 0
    for win, loss in scr:
        cursetnum += 1
        if cursetnum == setnum:
            break
        if win > loss:
            win_sets_count += 1
        elif win < loss:
            loss_sets_count += 1
    if cursetnum == 1:
        return "open", "open"
    if best_of_five is None:
        best_of_five = scr.best_of_five()
    fst_name, snd_name = "", ""
    if win_sets_count > loss_sets_count:
        fst_name, snd_name = "press", "under"
        if win_sets_count == 2 and loss_sets_count == 0 and best_of_five:
            fst_name, snd_name = "press2", "under2"
    elif win_sets_count < loss_sets_count:
        fst_name, snd_name = "under", "press"
        if win_sets_count == 0 and loss_sets_count == 2 and best_of_five:
            fst_name, snd_name = "under2", "press2"

    if cursetnum == 3 and win_sets_count == 1 and loss_sets_count == 1:
        if best_of_five:
            fst_name, snd_name = "open2", "open2"
        else:
            fst_name, snd_name = "decided", "decided"
    elif (
        cursetnum == 5 and win_sets_count == 2 and loss_sets_count == 2 and best_of_five
    ):
        fst_name, snd_name = "decided", "decided"

    if fst_name and snd_name:
        return fst_name, snd_name

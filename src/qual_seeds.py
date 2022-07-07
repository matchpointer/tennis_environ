import collections
from contextlib import closing
from typing import List, Dict

from oncourt import dbcon

# {tour_id -> List[player_id]}
__wta_quals_from_tid: Dict[int, List[int]] = collections.defaultdict(list)
__atp_quals_from_tid: Dict[int, List[int]] = collections.defaultdict(list)


def initialize():
    _initialize_sex("wta")
    _initialize_sex("atp")


def initialized():
    return __wta_quals_from_tid and __atp_quals_from_tid


def qual_players_idents(sex, tour_id):
    quals_from_tid = __wta_quals_from_tid if sex == "wta" else __atp_quals_from_tid
    return quals_from_tid[tour_id]


def _initialize_sex(sex):
    quals_from_tid = __wta_quals_from_tid if sex == "wta" else __atp_quals_from_tid
    sql = """select ID_T_S, ID_P_S
             from Seed_{0}
             where SEEDING Like '%q%' or SEEDING Like '%LL%'
             order by ID_T_S;""".format(
        sex
    )
    with closing(dbcon.get_connect().cursor()) as cursor:
        for (tour_id, player_id) in cursor.execute(sql):
            quals_from_tid[tour_id].append(player_id)


if __name__ == "__main__":
    from loguru import logger as log

    log.add('../log/qual_seeds.log', level='INFO',
            rotation='10:00', compression='zip')
    dbcon.open_connect()

    initialize()

    print("wta len: %d" % len(__wta_quals_from_tid))
    print("atp len: %d" % len(__atp_quals_from_tid))

    dbcon.close_connect()

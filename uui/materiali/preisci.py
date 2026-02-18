from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DrevesnoVozlisce:
    """Vozlišče v preiskovalnem drevesu.

    Pomembno:
    - To NI stanje v prostoru stanj, ampak konkretna pot od začetnega
        do tega stanja.
    - Isto stanje iz prostora stanj se lahko v drevesu pojavi večkrat
        (če do njega pridemo po različnih poteh).
    """
    s: Any     # stanje
    g: float   # cena poti od začetnega do tega vozlišča
    h: float   # hevristična ocena v stanju s
    stars: Optional["DrevesnoVozlisce"] = None   # kazalec na starša v drevesu
    poteza: Optional[Any] = None                 # poteza, ki vodi od starša do tega vozlišča
    otroci: List["DrevesnoVozlisce"] = field(default_factory=list)  # otroci v drevesu

    @property
    def f(self) -> float:
        return self.g + self.h


class Preisci:
    """Splošni preiskovalni algoritem + opcijsko grajenje preiskovalnega drevesa.

    ps je prostor stanj, objekt razreda z naslednjimi metodami:
      - ps.zacetno()    : vrne začetno stanje v prostoru stanj
      - ps.koncno(s)    : preveri, če je s končno stanje
      - ps.razveji(s)   : vrne seznam trojic (ns, cena, poteza); naslednje stanje, ceno poteze in njeno oznako
      - ps.h(s)         : vrne vrednost hevristike v stanju s

    Strategije:
      - "v-globino"           : Depth-First Search
      - "v-sirino"            : Breadth-First Search
      - "enotna-cena"         : Uniform-Cost Search (Dijkstra) -> min g
      - "najprej-najboljsi"   : Greedy best-first -> min h
      - "a*"                  : A* -> min (g + h)
    """

    def __init__(self, ps: Any, strategija: str, gradi_drevo: bool = True):
        self.ps = ps
        self.strategija = strategija
        self.gradi_drevo = gradi_drevo

        s0 = self.ps.zacetno()
        h0 = float(self.ps.h(s0))

        self.koren = DrevesnoVozlisce(s=s0, g=0.0, h=h0)
        self.vsa_vozlisca: List[DrevesnoVozlisce] = [self.koren]

        # Preiskovalna fronta (odprti seznam) drevesnih vozlišč
        self.odprti: List[DrevesnoVozlisce] = [self.koren]

        # Zaprti seznam je seznam že razvejanih stanj
        self.zaprti: set = set()

        # Najboljši g do posameznega stanja, pomembno za korektnost UCS/A*
        self.naj_g: Dict[Any, float] = {s0: 0.0}

        # Števec razvejanih vozlišč nastavimo na 0
        self.n_razvejana: int = 0

    # ---------------------------
    # Strategija izbire iz odprtega seznama
    # ---------------------------

    def _prioriteta(self, v: DrevesnoVozlisce) -> float:
        st = self.strategija
        if st in ("v-globino", "v-sirino"):
            return 0.0
        if st == "enotna-cena":
            return v.g
        if st == "najprej-najboljsi":
            return v.h
        if st == "a*":
            return v.f
        raise ValueError(f"Neznana strategija: {st!r}")

    def _vzemi_iz_odprtega(self) -> DrevesnoVozlisce:
        st = self.strategija
        if st == "v-sirino":   # first in, first out
            return self.odprti.pop(0)
        if st == "v-globino":  # last in, first out
            return self.odprti.pop()

        # izberi vozlišče z minimalno prioriteto
        i_min = min(range(len(self.odprti)), key=lambda i: self._prioriteta(self.odprti[i]))
        return self.odprti.pop(i_min)

    # ---------------------------
    # Rekonstrukcija poti iz DREVESNEGA vozlišča
    # ---------------------------

    def rekonstruiraj(self, cilj: DrevesnoVozlisce) -> Tuple[List[Any], List[Any]]:
        """Vrne (pot_stanj, pot_potez) od korena do vozlišča `cilj`."""
        stanja: List[Any] = []
        poteze: List[Any] = []

        # Sprehod po veji preiskovalnega drevesa navzgor do korena
        v: Optional[DrevesnoVozlisce] = cilj
        while v is not None:
            stanja.append(v.s)
            if v.poteza is not None:
                poteze.append(v.poteza)
            v = v.stars

        stanja.reverse()
        poteze.reverse()
        return stanja, poteze

    # ---------------------------
    # Glavni algoritem
    # ---------------------------

    def preisci(self) -> Optional[Tuple[List[Any], List[Any], float, int]]:
        """Vrne (pot_stanj, pot_potez, cena_g, st_razvejanih) ali None."""
        while True:
            # Preveri, če je zmanjkalo vozlišč na preiskovalni fronti
            if not self.odprti:
                return None

            v = self._vzemi_iz_odprtega()
            s, g = v.s, v.g

            # Če smo stanje s že razvejili, preskočimo
            if s in self.zaprti:
                continue

            # Če je stanje s končno, končamo preiskovanje
            if self.ps.koncno(s):
                stanja, poteze = self.rekonstruiraj(v)
                return (stanja, poteze, g, self.n_razvejana)

            # Sicer pa razvejimo stanje s
            self.n_razvejana += 1
            self.zaprti.add(s)

            for (ns, cena, poteza) in self.ps.razveji(s):
                ng = g + float(cena)

                # Posodobimo le, če je to boljša pot do stanja ns.
                if ng >= self.naj_g.get(ns, float("inf")):
                    continue

                self.naj_g[ns] = ng

                # Ustvarimo novo vozlišče preiskovalnega drevesa
                otrok = DrevesnoVozlisce(
                    s=ns,
                    g=ng,
                    h=float(self.ps.h(ns)),
                    stars=v,
                    poteza=poteza,
                )
                v.otroci.append(otrok)

                if self.gradi_drevo:
                    self.vsa_vozlisca.append(otrok)

                # Dodamo novo vozlišče v preiskovalno fronto (odprti seznam)
                self.odprti.append(otrok)

    def __str__(self) -> str:
        # Kratek izpis fronte: (stanje, g, h, f)
        odprti_str = [(v.s, v.g, v.h, v.f) for v in self.odprti]
        return (
            f"Strategija: {self.strategija}\n"
            f"Zaprti (|Z|={len(self.zaprti)}): {list(self.zaprti)}\n"
            f"Odprti (|O|={len(self.odprti)}): {odprti_str}\n"
            f"{self.n_razvejana} razvejanih vozlišč"
        )

"""
Haalt een TRAININGSSET van Mapillary-foto's uit Nederland op.

Zelfde werkwijze als mapillary_testset.py (willekeurige punten verspreid over
heel Nederland), maar met regels zodat de trainingsdata nooit "spiekt" bij de
testset:
  - geen foto's uit een opnamereeks (sequence) die ook in de testset zit;
  - geen foto's binnen 2 km van een testfoto;
  - maximaal één foto per opnamereeks, zodat er geen bijna-identieke
    foto's in zitten.
Per zoekvakje worden maximaal 3 foto's genomen (elk uit een andere reeks),
en de downloads lopen parallel, zodat het een stuk sneller gaat.

Benodigd: dezelfde token.txt en de map testset/ (voor de uitsluitingen).

Gebruik:
  python mapillary_trainset.py               # 5000 foto's
  python mapillary_trainset.py --aantal 10000

Je kunt het script stoppen (Ctrl+C) en later opnieuw starten; het gaat dan
verder. Wil je later meer data, start het dan met een hoger --aantal.
"""

import argparse
import csv
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

import mapillary_testset as mt

# ---------------- Instellingen ----------------
TRAIN_MAP = Path("trainset")
CSV_BESTAND = TRAIN_MAP / "trainset.csv"
MIN_AFSTAND_TEST_KM = 2.0       # nooit dichter bij een testfoto dan dit
PER_VAK = 3                     # max. foto's per zoekvakje
WERKERS = 8                     # zoveel zoekvragen/downloads tegelijk
SEED = 2026
KOLOMMEN = ["bestand", "image_id", "lat", "lon", "provincie",
            "datum", "sequence", "maker", "zoekpunt"]
# ----------------------------------------------


def afstand_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def lees_testset():
    if not mt.CSV_BESTAND.exists():
        sys.exit(f"{mt.CSV_BESTAND} niet gevonden: de testset is nodig om "
                 "overlap met de trainingsdata te voorkomen.")
    with open(mt.CSV_BESTAND, newline="", encoding="utf-8") as f:
        rijen = list(csv.DictReader(f))
    punten = [(float(r["lat"]), float(r["lon"])) for r in rijen]
    reeksen = {r["sequence"] for r in rijen}
    return punten, reeksen


def te_dicht_bij_test(lat, lon, testpunten):
    for tlat, tlon in testpunten:
        # snelle voorselectie, daarna de echte afstand
        if abs(lat - tlat) < 0.03 and abs(lon - tlon) < 0.05:
            if afstand_km(lat, lon, tlat, tlon) < MIN_AFSTAND_TEST_KM:
                return True
    return False


def download(foto):
    """Downloadt één foto; geeft (foto, bytes) of (foto, None) terug."""
    try:
        r = requests.get(foto["thumb_1024_url"], timeout=60)
        r.raise_for_status()
        return foto, r.content
    except requests.RequestException:
        return foto, None


def main():
    parser = argparse.ArgumentParser(description="Trainingsset ophalen van Mapillary")
    parser.add_argument("--aantal", type=int, default=5000,
                        help="hoeveel trainingsfoto's in totaal (standaard 5000)")
    args = parser.parse_args()

    token = mt.lees_token()
    provincies = mt.laad_provincies()
    testpunten, testreeksen = lees_testset()
    print(f"Testset: {len(testpunten)} foto's worden uitgesloten "
          f"(+ alles binnen {MIN_AFSTAND_TEST_KM:g} km).")

    nl_west = min(p["rechthoek"][0] for p in provincies)
    nl_zuid = min(p["rechthoek"][1] for p in provincies)
    nl_oost = max(p["rechthoek"][2] for p in provincies)
    nl_noord = max(p["rechthoek"][3] for p in provincies)

    TRAIN_MAP.mkdir(exist_ok=True)
    bekende_ids, bekende_reeksen = set(), set()
    nieuw_bestand = not CSV_BESTAND.exists()
    if not nieuw_bestand:
        with open(CSV_BESTAND, newline="", encoding="utf-8") as f:
            for rij in csv.DictReader(f):
                bekende_ids.add(rij["image_id"])
                bekende_reeksen.add(rij["sequence"])
        print(f"{len(bekende_ids)} trainingsfoto's al aanwezig, ga verder.")
    if len(bekende_ids) >= args.aantal:
        print("Al genoeg foto's.")
        return

    random.seed(SEED + len(bekende_ids))

    def willekeurig_punt():
        while True:
            lat = random.uniform(nl_zuid, nl_noord)
            lon = random.uniform(nl_west, nl_oost)
            if mt.zoek_provincie(lon, lat, provincies) is not None:
                return lat, lon

    def kies_fotos(fotos):
        """Kiest max. PER_VAK geschikte foto's uit één zoekresultaat."""
        random.shuffle(fotos)
        gekozen, reeksen_hier = [], set()
        for p in fotos:
            reeks = p.get("sequence")
            if (p.get("is_pano") or not p.get("thumb_1024_url")
                    or p["id"] in bekende_ids
                    or not reeks or reeks in bekende_reeksen
                    or reeks in testreeksen or reeks in reeksen_hier):
                continue
            lon, lat = p["geometry"]["coordinates"]
            provincie = mt.zoek_provincie(lon, lat, provincies)
            if provincie is None or te_dicht_bij_test(lat, lon, testpunten):
                continue
            p["_lat"], p["_lon"], p["_provincie"] = lat, lon, provincie
            gekozen.append(p)
            reeksen_hier.add(reeks)
            if len(gekozen) >= PER_VAK:
                break
        return gekozen

    pogingen = 0
    max_pogingen = args.aantal * 40
    volgende_melding = len(bekende_ids) + 100
    pool = ThreadPoolExecutor(max_workers=WERKERS)

    with open(CSV_BESTAND, "a", newline="", encoding="utf-8") as f:
        schrijver = csv.DictWriter(f, fieldnames=KOLOMMEN)
        if nieuw_bestand:
            schrijver.writeheader()
        try:
            while len(bekende_ids) < args.aantal and pogingen < max_pogingen:
                # 1. Een groepje zoekvakjes tegelijk opvragen
                punten = [willekeurig_punt() for _ in range(WERKERS)]
                pogingen += len(punten)
                resultaten = list(pool.map(lambda pt: mt.veilig_zoeken(token, pt), punten))

                # 2. Per vakje geschikte foto's kiezen
                ruimte = args.aantal - len(bekende_ids)
                te_downloaden = []
                for (plat, plon), fotos in zip(punten, resultaten):
                    for foto in kies_fotos(fotos):
                        if len(te_downloaden) >= ruimte:
                            break
                        foto["_zoekpunt"] = f"{plat:.4f},{plon:.4f}"
                        te_downloaden.append(foto)
                        bekende_ids.add(foto["id"])          # alvast reserveren
                        bekende_reeksen.add(foto["sequence"])

                # 3. Parallel downloaden en wegschrijven
                for foto, inhoud in pool.map(download, te_downloaden):
                    if inhoud is None:
                        bekende_ids.discard(foto["id"])
                        continue
                    bestandsnaam = f"{foto['id']}.jpg"
                    (TRAIN_MAP / bestandsnaam).write_bytes(inhoud)
                    datum = ""
                    if foto.get("captured_at"):
                        datum = datetime.fromtimestamp(
                            foto["captured_at"] / 1000, tz=timezone.utc).date().isoformat()
                    schrijver.writerow({
                        "bestand": bestandsnaam, "image_id": foto["id"],
                        "lat": f"{foto['_lat']:.6f}", "lon": f"{foto['_lon']:.6f}",
                        "provincie": foto["_provincie"], "datum": datum,
                        "sequence": foto["sequence"],
                        "maker": (foto.get("creator") or {}).get("username", ""),
                        "zoekpunt": foto["_zoekpunt"],
                    })
                f.flush()

                if len(bekende_ids) >= volgende_melding or len(bekende_ids) >= args.aantal:
                    print(f"[{len(bekende_ids)}/{args.aantal}] trainingsfoto's "
                          f"({pogingen} plekken bekeken)")
                    volgende_melding = len(bekende_ids) + 100
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    print(f"\nKlaar: {len(bekende_ids)} foto's in '{TRAIN_MAP}/', "
          f"gegevens in '{CSV_BESTAND}'.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGestopt. Start het script opnieuw om verder te gaan.")

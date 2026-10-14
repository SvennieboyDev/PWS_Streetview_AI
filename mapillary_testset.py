"""
Haalt willekeurige Mapillary-foto's uit Nederland op, voor een testset.

Werkwijze:
  1. Kies een willekeurig punt in Nederland (gecontroleerd met de echte
     provinciegrenzen van CBS/PDOK, dus geen zee, België of Duitsland).
  2. Vraag bij Mapillary de foto's op in een klein vakje rond dat punt.
  3. Kies daar één willekeurige (niet-panorama) foto uit, download hem
     en schrijf de echte coördinaten + provincie naar testset.csv.

Benodigd (alleen deze ene extra package):
  pip install requests

Token:
  Zet je Mapillary client token (begint met MLY|) in een bestand
  token.txt naast dit script, of in de omgevingsvariabele MAPILLARY_TOKEN.
  Zet token.txt nooit op GitHub!

Gebruik:
  python mapillary_testset.py

De eerste keer downloadt het script de provinciegrenzen (provincies.geojson).
Je kunt het script onderbreken (Ctrl+C) en later opnieuw starten:
het gaat dan verder waar het gebleven was.
"""

import csv
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------- Instellingen ----------------
AANTAL_FOTOS = 250              # hoeveel foto's je in de testset wilt
UITVOER_MAP = Path("testset")   # hier komen de foto's en de CSV
CSV_BESTAND = UITVOER_MAP / "testset.csv"
TOKEN_BESTAND = Path("token.txt")
GRENZEN_BESTAND = Path("provincies.geojson")
SEED = 42                       # vaste startwaarde, zodat de puntkeuze herhaalbaar is

BOX_GROOTTE = 0.04              # zoekvakje van ca. 2,7 km (oost-west) x 4,4 km (noord-zuid)
MAX_POGINGEN = 20000            # veiligheidsgrens zodat het script nooit eindeloos blijft lopen
WERKERS = 8                     # zoveel zoekvragen tegelijk naar Mapillary

# Provinciegrenzen van CBS, via PDOK (coördinaten in lengte-/breedtegraad)
GRENZEN_URL = (
    "https://service.pdok.nl/cbs/gebiedsindelingen/2023/wfs/v1_0"
    "?request=GetFeature&service=WFS&version=2.0.0"
    "&typeName=gebiedsindelingen:provincie_gegeneraliseerd"
    "&outputFormat=json&srsName=EPSG:4326"
)

API_URL = "https://graph.mapillary.com/images"
VELDEN = "id,geometry,thumb_1024_url,captured_at,is_pano,sequence,creator"
KOLOMMEN = ["bestand", "image_id", "lat", "lon", "provincie",
            "datum", "sequence", "maker"]
# ----------------------------------------------


def lees_token():
    token = os.environ.get("MAPILLARY_TOKEN")
    if not token and TOKEN_BESTAND.exists():
        token = TOKEN_BESTAND.read_text(encoding="utf-8").strip()
    if not token:
        sys.exit("Geen token gevonden. Zet je token (MLY|...) in token.txt naast dit script.")
    return token


# ---------- Provinciegrenzen ----------

def laad_provincies():
    """Downloadt (eenmalig) de provinciegrenzen en geeft een lijst terug met
    per provincie: naam, omhullende rechthoek en de polygonen."""
    if not GRENZEN_BESTAND.exists():
        print("Provinciegrenzen downloaden van PDOK...")
        r = requests.get(GRENZEN_URL, timeout=60)
        r.raise_for_status()
        GRENZEN_BESTAND.write_text(r.text, encoding="utf-8")

    data = json.loads(GRENZEN_BESTAND.read_text(encoding="utf-8"))
    provincies = []
    for feature in data["features"]:
        geo = feature["geometry"]
        polygonen = geo["coordinates"] if geo["type"] == "MultiPolygon" else [geo["coordinates"]]
        alle_punten = [p for poly in polygonen for ring in poly for p in ring]
        lons = [p[0] for p in alle_punten]
        lats = [p[1] for p in alle_punten]
        provincies.append({
            "naam": feature["properties"]["statnaam"],
            "rechthoek": (min(lons), min(lats), max(lons), max(lats)),
            "polygonen": polygonen,
        })
    return provincies


def punt_in_ring(lon, lat, ring):
    """Ray casting: tel hoe vaak een lijn vanaf het punt de rand kruist.
    Oneven aantal kruisingen = punt ligt binnen de ring."""
    binnen = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat):
            x_kruising = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_kruising:
                binnen = not binnen
        j = i
    return binnen


def zoek_provincie(lon, lat, provincies):
    """Geeft de provincienaam terug, of None als het punt niet in Nederland ligt."""
    for prov in provincies:
        west, zuid, oost, noord = prov["rechthoek"]
        if not (west <= lon <= oost and zuid <= lat <= noord):
            continue
        for polygoon in prov["polygonen"]:
            buitenrand, gaten = polygoon[0], polygoon[1:]
            if punt_in_ring(lon, lat, buitenrand) and not any(
                punt_in_ring(lon, lat, gat) for gat in gaten
            ):
                return prov["naam"]
    return None


# ---------- Mapillary ----------

class MapillaryFout(Exception):
    pass


def zoek_fotos(token, lat, lon):
    """Vraagt foto's op rond een punt. Geeft Mapillary een serverfout (500),
    wat vooral in drukke gebieden gebeurt, dan wordt het vakje steeds
    kleiner gemaakt en opnieuw geprobeerd."""
    laatste_fout = None
    for grootte in (BOX_GROOTTE, BOX_GROOTTE / 2, BOX_GROOTTE / 4):
        half = grootte / 2
        # Volgorde van bbox: west, zuid, oost, noord
        bbox = f"{lon - half:.5f},{lat - half:.5f},{lon + half:.5f},{lat + half:.5f}"
        params = {"access_token": token, "fields": VELDEN, "bbox": bbox, "limit": 50}
        try:
            r = requests.get(API_URL, params=params, timeout=30)
        except requests.RequestException as e:
            # Alleen het soort fout tonen, niet de URL: daar staat je token in
            raise MapillaryFout(type(e).__name__) from None
        if r.status_code in (401, 403):
            sys.exit(f"Mapillary weigert de token ({r.status_code}). Controleer token.txt.")
        if r.status_code >= 500:
            laatste_fout = f"serverfout {r.status_code}"
            continue  # opnieuw proberen met een kleiner vakje
        if r.status_code != 200:
            raise MapillaryFout(f"statuscode {r.status_code}")
        return r.json().get("data", [])
    raise MapillaryFout(laatste_fout)


def veilig_zoeken(token, punt):
    """Zoekt foto's rond een punt; lukt dat niet, dan gewoon een lege lijst."""
    lat, lon = punt
    try:
        return zoek_fotos(token, lat, lon)
    except MapillaryFout as e:
        print(f"  Mapillary gaf een fout ({e}) bij {lat:.3f}, {lon:.3f}, overslaan.")
        return []


def main():
    token = lees_token()
    provincies = laad_provincies()
    nl_west = min(p["rechthoek"][0] for p in provincies)
    nl_zuid = min(p["rechthoek"][1] for p in provincies)
    nl_oost = max(p["rechthoek"][2] for p in provincies)
    nl_noord = max(p["rechthoek"][3] for p in provincies)

    UITVOER_MAP.mkdir(exist_ok=True)

    # Eerder opgehaalde foto's inlezen, zodat we verder kunnen waar we waren
    bekende_ids, bekende_sequences = set(), set()
    nieuw_bestand = not CSV_BESTAND.exists()
    if not nieuw_bestand:
        with open(CSV_BESTAND, newline="", encoding="utf-8") as f:
            for rij in csv.DictReader(f):
                bekende_ids.add(rij["image_id"])
                bekende_sequences.add(rij["sequence"])
        print(f"{len(bekende_ids)} foto's al aanwezig, ga verder.")

    random.seed(SEED + len(bekende_ids))

    with open(CSV_BESTAND, "a", newline="", encoding="utf-8") as f:
        schrijver = csv.DictWriter(f, fieldnames=KOLOMMEN)
        if nieuw_bestand:
            schrijver.writeheader()

        def willekeurig_punt():
            """Willekeurig punt dat in Nederland ligt."""
            while True:
                lat = random.uniform(nl_zuid, nl_noord)
                lon = random.uniform(nl_west, nl_oost)
                if zoek_provincie(lon, lat, provincies) is not None:
                    return lat, lon

        pogingen = 0
        volgende_melding = 100
        pool = ThreadPoolExecutor(max_workers=WERKERS)
        try:
            while len(bekende_ids) < AANTAL_FOTOS and pogingen < MAX_POGINGEN:
                # 1 + 2. Een groepje punten tegelijk opvragen bij Mapillary
                punten = [willekeurig_punt() for _ in range(WERKERS)]
                pogingen += len(punten)
                resultaten = list(pool.map(lambda p: veilig_zoeken(token, p), punten))

                if pogingen >= volgende_melding:
                    print(f"  ... {pogingen} plekken bekeken, "
                          f"{len(bekende_ids)} foto's gevonden")
                    volgende_melding += 100

                for fotos in resultaten:
                    if len(bekende_ids) >= AANTAL_FOTOS:
                        break
                    if verwerk(fotos, provincies, bekende_ids,
                               bekende_sequences, schrijver):
                        f.flush()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    print(f"\nKlaar: {len(bekende_ids)} foto's in '{UITVOER_MAP}/', "
          f"gegevens in '{CSV_BESTAND}'.")


def verwerk(fotos, provincies, bekende_ids, bekende_sequences, schrijver):
    """Kiest één foto uit de zoekresultaten, downloadt hem en schrijft een
    regel naar de CSV. Geeft True terug als dat gelukt is."""
    # 3. Geen panorama's, en maximaal één foto per reeks (sequence),
    #    zodat je geen bijna-identieke foto's krijgt
    fotos = [
        p for p in fotos
        if not p.get("is_pano")
        and p.get("thumb_1024_url")
        and p["id"] not in bekende_ids
        and p.get("sequence") not in bekende_sequences
    ]
    if not fotos:
        return False

    foto = random.choice(fotos)
    foto_lon, foto_lat = foto["geometry"]["coordinates"]
    provincie = zoek_provincie(foto_lon, foto_lat, provincies)
    if provincie is None:
        return False  # foto ligt net over de grens

    # 4. Foto downloaden
    bestandsnaam = f"{foto['id']}.jpg"
    try:
        afbeelding = requests.get(foto["thumb_1024_url"], timeout=60)
        afbeelding.raise_for_status()
    except requests.RequestException as e:
        print(f"  Download mislukt ({e}), overslaan.")
        return False
    (UITVOER_MAP / bestandsnaam).write_bytes(afbeelding.content)

    datum = ""
    if foto.get("captured_at"):
        datum = datetime.fromtimestamp(
            foto["captured_at"] / 1000, tz=timezone.utc
        ).date().isoformat()

    schrijver.writerow({
        "bestand": bestandsnaam,
        "image_id": foto["id"],
        "lat": f"{foto_lat:.6f}",
        "lon": f"{foto_lon:.6f}",
        "provincie": provincie,
        "datum": datum,
        "sequence": foto.get("sequence", ""),
        "maker": (foto.get("creator") or {}).get("username", ""),
    })

    bekende_ids.add(foto["id"])
    bekende_sequences.add(foto.get("sequence", ""))
    print(f"[{len(bekende_ids)}/{AANTAL_FOTOS}] {provincie}  "
          f"({foto_lat:.4f}, {foto_lon:.4f})")
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGestopt. Start het script opnieuw om verder te gaan.")
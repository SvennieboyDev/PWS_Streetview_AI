"""
Test bestaande geolocatiemodellen op de Mapillary-testset (haalbaarheidstoets).

Methoden die worden vergeleken:
  GeoCLIP (wereld)     - kant-en-klaar GeoCLIP, kiest uit 100.000 plekken op
                          de hele wereld.
  GeoCLIP (alleen NL)  - hetzelfde model, maar mag alleen kiezen uit een raster
                          van punten in Nederland (elke ~2 km). Zo zie je wat
                          het model kan als het al weet dat de foto in NL is.
  StreetCLIP           - kiest welke van de 342 gemeenten het best bij de foto
                          past; de voorspelling is het midden van die gemeente.
  Middelpunt           - domme baseline: altijd het midden van de testset.
  Willekeurig punt     - domme baseline: een willekeurig punt in Nederland.

Uitvoer (map 'resultaten/'):
  samenvatting.csv     - per methode: mediane fout, % binnen 1/5/10/25/50 km,
                          % juiste provincie
  voorspellingen.csv   - per foto per methode: echte plek, voorspelling, fout
  fout_cumulatief.png  - grafiek: % foto's binnen X km, per methode
  kaart_fouten.png     - kaart met lijntjes van echte plek naar voorspelling

Benodigd:
  pip install geoclip matplotlib requests
  (geoclip installeert zelf torch en transformers)

Gebruik (in dezelfde map als mapillary_testset.py en de map testset/):
  python evalueer_modellen.py              # alles
  python evalueer_modellen.py --max 10     # snelle proef met 10 foto's
  python evalueer_modellen.py --zonder-streetclip

De eerste keer worden de modellen gedownload (elk ca. 1,7 GB). Op een laptop
zonder grafische kaart duurt het ongeveer 2-5 seconden per foto per model.
Voorspellingen worden tussendoor opgeslagen: stop je het script, dan gaat
het de volgende keer verder waar het was.
"""

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path

import requests

from mapillary_testset import CSV_BESTAND, UITVOER_MAP as TESTSET_MAP
from mapillary_testset import laad_provincies, punt_in_ring, zoek_provincie

# ---------------- Instellingen ----------------
RESULTATEN = Path("resultaten")
GEMEENTEN_BESTAND = Path("gemeenten.geojson")
GEMEENTEN_URL = (
    "https://service.pdok.nl/cbs/gebiedsindelingen/2023/wfs/v1_0"
    "?request=GetFeature&service=WFS&version=2.0.0"
    "&typeName=gebiedsindelingen:gemeente_gegeneraliseerd"
    "&outputFormat=json&srsName=EPSG:4326"
)
RASTER_KM = 2                   # afstand tussen rasterpunten voor GeoCLIP (alleen NL)
DREMPELS_KM = [1, 5, 10, 25, 50]
# Tekstsjabloon uit het StreetCLIP-artikel
STREETCLIP_ZIN = ("A Street View photo close to the town of {gemeente} "
                  "in the region of {provincie} in Netherlands.")
# ----------------------------------------------


# ---------- Hulpfuncties ----------

def afstand_km(lat1, lon1, lat2, lon2):
    """Haversine-formule: afstand over het aardoppervlak in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def lees_testset(maximum=None):
    if not CSV_BESTAND.exists():
        sys.exit(f"{CSV_BESTAND} niet gevonden. Draai eerst mapillary_testset.py.")
    with open(CSV_BESTAND, newline="", encoding="utf-8") as f:
        fotos = [
            {**rij, "lat": float(rij["lat"]), "lon": float(rij["lon"])}
            for rij in csv.DictReader(f)
            if (TESTSET_MAP / rij["bestand"]).exists()
        ]
    return fotos[:maximum] if maximum else fotos


def provincie_van(lat, lon, provincies):
    return zoek_provincie(lon, lat, provincies) or "buiten NL"


def zwaartepunt(ring):
    """Zwaartepunt van een polygoon (lon, lat)."""
    a = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        k = x1 * y2 - x2 * y1
        a += k
        cx += (x1 + x2) * k
        cy += (y1 + y2) * k
    if abs(a) < 1e-12:
        return sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring)
    return cx / (3 * a), cy / (3 * a)


def ring_oppervlak(ring):
    return abs(sum(x1 * y2 - x2 * y1
                   for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]))) / 2


def laad_gemeenten(provincies):
    """Gemeenten met hun middelpunt en provincie (voor StreetCLIP)."""
    if not GEMEENTEN_BESTAND.exists():
        print("Gemeentegrenzen downloaden van PDOK...")
        r = requests.get(GEMEENTEN_URL, timeout=60)
        r.raise_for_status()
        GEMEENTEN_BESTAND.write_text(r.text, encoding="utf-8")
    data = json.loads(GEMEENTEN_BESTAND.read_text(encoding="utf-8"))
    gemeenten = []
    for feature in data["features"]:
        geo = feature["geometry"]
        polygonen = geo["coordinates"] if geo["type"] == "MultiPolygon" else [geo["coordinates"]]
        # Midden van het grootste deel (bij gemeenten met eilanden)
        ring = [tuple(p[:2]) for p in max((p[0] for p in polygonen), key=ring_oppervlak)]
        lon, lat = zwaartepunt(ring)
        provincie = zoek_provincie(lon, lat, provincies)
        if provincie is None:  # midden valt net buiten de grens (bijv. in zee)
            provincie = zoek_provincie(ring[0][0], ring[0][1], provincies) or ""
        gemeenten.append({"naam": feature["properties"]["statnaam"],
                          "provincie": provincie, "lat": lat, "lon": lon})
    return gemeenten


def nl_raster(provincies, stap_km=RASTER_KM):
    """Raster van punten (lat, lon) binnen Nederland."""
    west = min(p["rechthoek"][0] for p in provincies)
    zuid = min(p["rechthoek"][1] for p in provincies)
    oost = max(p["rechthoek"][2] for p in provincies)
    noord = max(p["rechthoek"][3] for p in provincies)
    dlat = stap_km / 111.0
    dlon = stap_km / (111.0 * math.cos(math.radians(52.2)))
    punten = []
    lat = zuid
    while lat <= noord:
        lon = west
        while lon <= oost:
            if zoek_provincie(lon, lat, provincies) is not None:
                punten.append((lat, lon))
            lon += dlon
        lat += dlat
    return punten


def kies_apparaat():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def als_tensor(uitvoer):
    """CLIP geeft afhankelijk van de transformers-versie een tensor of een
    object terug; dit haalt er altijd de tensor uit."""
    import torch
    if isinstance(uitvoer, torch.Tensor):
        return uitvoer
    pooled = getattr(uitvoer, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    return uitvoer[0]


# ---------- Tussentijds opslaan ----------

def lees_cache(naam):
    pad = RESULTATEN / f"cache_{naam}.json"
    return json.loads(pad.read_text(encoding="utf-8")) if pad.exists() else {}


def schrijf_cache(naam, cache):
    (RESULTATEN / f"cache_{naam}.json").write_text(
        json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def draai_model(naam, fotos, maak_model):
    """Laat een model alle foto's voorspellen, met tussentijds opslaan.
    maak_model() wordt alleen aangeroepen als er nog iets te doen is."""
    cache = lees_cache(naam)
    te_doen = [f for f in fotos if f["bestand"] not in cache]
    if not te_doen:
        print(f"{naam}: alle voorspellingen al aanwezig.")
        return cache
    print(f"\n{naam}: model laden...")
    model = maak_model()
    for i, foto in enumerate(te_doen, 1):
        cache[foto["bestand"]] = model.voorspel(TESTSET_MAP / foto["bestand"])
        if i % 10 == 0 or i == len(te_doen):
            schrijf_cache(naam, cache)
            print(f"  {naam}: {len(cache)}/{len(fotos)} foto's gedaan")
    del model
    return cache


# ---------- Modellen ----------

class GeoCLIPVoorspeller:
    """Eén keer de foto door het model; daarna vergelijken met zowel de
    wereldgalerij als het Nederlandse raster."""

    def __init__(self, provincies):
        import torch
        import torch.nn.functional as F
        from geoclip import GeoCLIP
        self.torch, self.F = torch, F
        self.apparaat = kies_apparaat()
        self.model = self.maak_geoclip().to(self.apparaat).eval()

        self.wereld = self.model.gps_gallery.to(self.apparaat)
        raster = nl_raster(provincies)
        print(f"  raster met {len(raster)} punten in Nederland")
        self.nl = torch.tensor(raster, dtype=torch.float32, device=self.apparaat)
        self.wereld_vec = self.plek_vectoren(self.wereld)
        self.nl_vec = self.plek_vectoren(self.nl)

    def maak_geoclip(self):
        from geoclip import GeoCLIP
        return GeoCLIP()

    def plek_vectoren(self, gps):
        with self.torch.no_grad():
            delen = [self.model.location_encoder(gps[i:i + 10000])
                     for i in range(0, len(gps), 10000)]
        return self.F.normalize(self.torch.cat(delen), dim=1)

    def voorspel(self, pad):
        from PIL import Image
        with Image.open(pad) as img:
            x = self.model.image_encoder.preprocess_image(img.convert("RGB"))
        with self.torch.no_grad():
            v = self.F.normalize(self.model.image_encoder(x.to(self.apparaat)), dim=1)
        wereld = self.wereld[(v @ self.wereld_vec.T).argmax()].tolist()
        nl = self.nl[(v @ self.nl_vec.T).argmax()].tolist()
        return {"wereld": wereld, "nl": nl}


class StreetCLIPVoorspeller:
    def __init__(self, gemeenten):
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        self.apparaat = kies_apparaat()
        self.model, self.processor = self.maak_streetclip()
        self.model = self.model.to(self.apparaat).eval()
        self.gemeenten = gemeenten

        zinnen = [STREETCLIP_ZIN.format(gemeente=g["naam"], provincie=g["provincie"])
                  for g in gemeenten]
        delen = []
        with torch.no_grad():
            for i in range(0, len(zinnen), 32):
                invoer = self.processor(text=zinnen[i:i + 32], return_tensors="pt",
                                        padding=True).to(self.apparaat)
                delen.append(als_tensor(self.model.get_text_features(**invoer)))
        self.tekst_vec = F.normalize(torch.cat(delen), dim=1)

    def maak_streetclip(self):
        from transformers import CLIPModel, CLIPProcessor
        return (CLIPModel.from_pretrained("geolocal/StreetCLIP"),
                CLIPProcessor.from_pretrained("geolocal/StreetCLIP"))

    def voorspel(self, pad):
        from PIL import Image
        with Image.open(pad) as img:
            invoer = self.processor(images=img.convert("RGB"), return_tensors="pt")
        with self.torch.no_grad():
            v = als_tensor(self.model.get_image_features(
                pixel_values=invoer["pixel_values"].to(self.apparaat)))
        v = self.F.normalize(v, dim=1)
        g = self.gemeenten[int((v @ self.tekst_vec.T).argmax())]
        return {"gemeente": g["naam"], "provincie": g["provincie"],
                "lat": g["lat"], "lon": g["lon"]}


# ---------- Evalueren ----------

def maak_voorspellingen(fotos, provincies, args):
    """Geeft {methode: [(lat, lon, provincie, extra), ...]} terug,
    in dezelfde volgorde als fotos."""
    methoden = {}

    geo = draai_model("geoclip", fotos, lambda: GeoCLIPVoorspeller(provincies))
    for sleutel, naam in (("wereld", "GeoCLIP (wereld)"), ("nl", "GeoCLIP (alleen NL)")):
        methoden[naam] = []
        for f in fotos:
            lat, lon = geo[f["bestand"]][sleutel]
            methoden[naam].append((lat, lon, provincie_van(lat, lon, provincies), ""))

    if not args.zonder_streetclip:
        gemeenten = laad_gemeenten(provincies)
        sc = draai_model("streetclip", fotos, lambda: StreetCLIPVoorspeller(gemeenten))
        methoden["StreetCLIP"] = [
            (sc[f["bestand"]]["lat"], sc[f["bestand"]]["lon"],
             sc[f["bestand"]]["provincie"], sc[f["bestand"]]["gemeente"])
            for f in fotos
        ]

    # Domme baselines
    mid_lat = statistics.mean(f["lat"] for f in fotos)
    mid_lon = statistics.mean(f["lon"] for f in fotos)
    mid_prov = provincie_van(mid_lat, mid_lon, provincies)
    methoden["Middelpunt (baseline)"] = [(mid_lat, mid_lon, mid_prov, "")] * len(fotos)

    rng = random.Random(1)
    raster = nl_raster(provincies, stap_km=1)
    methoden["Willekeurig punt (baseline)"] = []
    for _ in fotos:
        lat, lon = rng.choice(raster)
        methoden["Willekeurig punt (baseline)"].append(
            (lat, lon, provincie_van(lat, lon, provincies), ""))
    return methoden


def samenvatten(fotos, methoden):
    regels, details = [], []
    for naam, voorsp in methoden.items():
        fouten = []
        goed_prov = 0
        for f, (lat, lon, prov, extra) in zip(fotos, voorsp):
            fout = afstand_km(f["lat"], f["lon"], lat, lon)
            fouten.append(fout)
            goed_prov += prov == f["provincie"]
            details.append({
                "bestand": f["bestand"], "methode": naam,
                "echte_lat": f["lat"], "echte_lon": f["lon"],
                "echte_provincie": f["provincie"],
                "voorspelde_lat": round(lat, 6), "voorspelde_lon": round(lon, 6),
                "voorspelde_provincie": prov, "voorspelde_gemeente": extra,
                "fout_km": round(fout, 2),
            })
        regel = {"methode": naam, "n": len(fouten),
                 "mediane_fout_km": round(statistics.median(fouten), 1),
                 "gemiddelde_fout_km": round(statistics.mean(fouten), 1)}
        for d in DREMPELS_KM:
            regel[f"binnen_{d}km_%"] = round(100 * sum(x <= d for x in fouten) / len(fouten), 1)
        regel["juiste_provincie_%"] = round(100 * goed_prov / len(fouten), 1)
        regel["_fouten"] = fouten
        regels.append(regel)
    return regels, details


def schrijf_csv(pad, rijen):
    with open(pad, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rijen[0] if not k.startswith("_")],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rijen)


def print_tabel(regels):
    kop = ["methode", "mediaan", "≤1km", "≤5km", "≤10km", "≤25km", "≤50km", "provincie"]
    print("\n" + f"{kop[0]:<28}" + "".join(f"{k:>10}" for k in kop[1:]))
    print("-" * (28 + 10 * (len(kop) - 1)))
    for r in regels:
        waarden = [f"{r['mediane_fout_km']} km"] + \
                  [f"{r[f'binnen_{d}km_%']}%" for d in DREMPELS_KM] + \
                  [f"{r['juiste_provincie_%']}%"]
        print(f"{r['methode']:<28}" + "".join(f"{w:>10}" for w in waarden))


# ---------- Grafieken ----------

# Kleuren: modellen in duidelijke kleuren, baselines in grijs
KLEUREN = {
    "GeoCLIP (wereld)": "#2a78d6",
    "GeoCLIP (alleen NL)": "#eb6834",
    "StreetCLIP": "#1baf7a",
}
BASELINE_STIJL = {
    "Middelpunt (baseline)": ("#8a8983", (0, (5, 3))),
    "Willekeurig punt (baseline)": ("#8a8983", (0, (1, 2))),
}
TEKST, TEKST_2, RAND, OPPERVLAK = "#0b0b0b", "#52514e", "#d9d8d3", "#fcfcfb"


def stijl_assen(ax):
    ax.set_facecolor(OPPERVLAK)
    for kant in ("top", "right"):
        ax.spines[kant].set_visible(False)
    for kant in ("left", "bottom"):
        ax.spines[kant].set_color(RAND)
    ax.tick_params(colors=TEKST_2, labelsize=9)


def grafiek_cumulatief(regels, pad):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150, facecolor=OPPERVLAK)
    stijl_assen(ax)
    xs = [10 ** (i / 50) for i in range(-25, 126)]  # 0,3 km tot 316 km
    for r in regels:
        fouten = sorted(r["_fouten"])
        ys = [100 * sum(f <= x for f in fouten) / len(fouten) for x in xs]
        if r["methode"] in KLEUREN:
            ax.plot(xs, ys, color=KLEUREN[r["methode"]], lw=2, label=r["methode"])
        elif r["methode"] in BASELINE_STIJL:
            kleur, streep = BASELINE_STIJL[r["methode"]]
            ax.plot(xs, ys, color=kleur, lw=1.6, linestyle=streep, label=r["methode"])

    ax.set_xscale("log")
    ticks = [0.5, 1, 2, 5, 10, 25, 50, 100, 250]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, 100)
    ax.grid(True, color=RAND, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Afstand tot de echte plek (km, logaritmische schaal)", color=TEKST_2, fontsize=10)
    ax.set_ylabel("Foto's binnen die afstand (%)", color=TEKST_2, fontsize=10)
    ax.set_title("Hoe dichtbij zit de voorspelling?", color=TEKST, fontsize=13,
                 loc="left", pad=12, fontweight="bold")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TEKST)
    fig.tight_layout()
    fig.savefig(pad, facecolor=OPPERVLAK, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def kaart_fouten(fotos, methoden, regels, provincies, pad):
    import matplotlib.pyplot as plt

    modellen = [m for m in methoden if m in KLEUREN]
    mediaan = {r["methode"]: r["mediane_fout_km"] for r in regels}
    west = min(p["rechthoek"][0] for p in provincies) - 0.1
    zuid = min(p["rechthoek"][1] for p in provincies) - 0.1
    oost = max(p["rechthoek"][2] for p in provincies) + 0.1
    noord = max(p["rechthoek"][3] for p in provincies) + 0.1

    # 1-3 kaarten naast elkaar, 4 als 2x2, meer in rijen van 3
    kolommen = len(modellen) if len(modellen) <= 3 else (2 if len(modellen) == 4 else 3)
    rijen = math.ceil(len(modellen) / kolommen)
    fig, assen = plt.subplots(rijen, kolommen, figsize=(5.2 * kolommen, 6.3 * rijen + 0.9),
                              dpi=150, facecolor=OPPERVLAK, squeeze=False)
    for ax in assen.flat[len(modellen):]:
        ax.set_visible(False)  # lege plekken in het raster
    for ax, naam in zip(assen.flat, modellen):
        ax.set_facecolor(OPPERVLAK)
        for prov in provincies:
            for poly in prov["polygonen"]:
                ring = poly[0]
                ax.plot([p[0] for p in ring], [p[1] for p in ring],
                        color="#c9c8c2", lw=0.6, zorder=1)
        buiten = 0
        for f, (lat, lon, _, _) in zip(fotos, methoden[naam]):
            if not (west <= lon <= oost and zuid <= lat <= noord):
                buiten += 1
            ax.plot([f["lon"], lon], [f["lat"], lat], color=KLEUREN[naam],
                    lw=0.6, alpha=0.35, zorder=2)
        ax.scatter([f["lon"] for f in fotos], [f["lat"] for f in fotos],
                   s=8, color=TEKST_2, zorder=3, linewidths=0)
        ax.scatter([v[1] for v in methoden[naam]], [v[0] for v in methoden[naam]],
                   s=14, color=KLEUREN[naam], zorder=4,
                   edgecolors=OPPERVLAK, linewidths=0.6)
        ax.set_xlim(west, oost)
        ax.set_ylim(zuid, noord)
        ax.set_aspect(1 / math.cos(math.radians(52.2)))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        # Titel per kaart: modelnaam vet, daaronder de mediane fout
        ax.text(0.0, 1.07, naam, transform=ax.transAxes, color=TEKST,
                fontsize=12, fontweight="bold", va="bottom")
        ax.text(0.0, 1.02, f"mediane fout {mediaan[naam]} km", transform=ax.transAxes,
                color=TEKST_2, fontsize=10, va="bottom")
        if buiten:
            ax.text(0.0, -0.02, f"{buiten} voorspelling(en) buiten beeld",
                    transform=ax.transAxes, color=TEKST_2, fontsize=8, va="top")

    # Eén uitleg en één legenda voor de hele figuur
    from matplotlib.lines import Line2D
    kop = 0.9 / (6.3 * rijen + 0.9)  # deel van de hoogte voor titel, uitleg en legenda
    fig.suptitle("Waar legt elk model de testfoto's neer?", x=0.01, ha="left",
                 color=TEKST, fontsize=14, fontweight="bold", y=1.0)
    fig.text(0.01, 1 - kop * 0.5, "Elk lijntje loopt van de echte plek van een foto "
             "(grijze stip) naar de plek die het model voorspelde (gekleurde stip). "
             "Korte lijntjes = goed.", ha="left", va="top", color=TEKST_2, fontsize=10)
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color=TEKST_2, markersize=4, label="echte plek"),
        Line2D([], [], marker="o", ls="", color="#8a8983", markersize=5,
               label="voorspelling (kleur van het model)"),
    ], loc="upper left", frameon=False, fontsize=9, ncol=2,
        bbox_to_anchor=(0.003, 1 - kop * 0.78))
    # h_pad: ruimte tussen de rijen, zodat titels niet over de kaart erboven vallen
    fig.tight_layout(rect=(0, 0, 1, 1 - kop * 1.4), h_pad=5)
    # bbox_inches="tight" zorgt dat er nooit tekst van de rand valt
    fig.savefig(pad, facecolor=OPPERVLAK, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--max", type=int, help="alleen de eerste N foto's (snelle proef)")
    parser.add_argument("--zonder-streetclip", action="store_true",
                        help="StreetCLIP overslaan")
    args = parser.parse_args()

    RESULTATEN.mkdir(exist_ok=True)
    provincies = laad_provincies()
    fotos = lees_testset(args.max)
    print(f"{len(fotos)} foto's in de testset.")

    methoden = maak_voorspellingen(fotos, provincies, args)
    regels, details = samenvatten(fotos, methoden)

    schrijf_csv(RESULTATEN / "samenvatting.csv", regels)
    schrijf_csv(RESULTATEN / "voorspellingen.csv", details)
    print_tabel(regels)

    import matplotlib
    matplotlib.use("Agg")
    grafiek_cumulatief(regels, RESULTATEN / "fout_cumulatief.png")
    kaart_fouten(fotos, methoden, regels, provincies, RESULTATEN / "kaart_fouten.png")
    print(f"\nKlaar. Alles staat in '{RESULTATEN}/'.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGestopt. Start opnieuw om verder te gaan.")

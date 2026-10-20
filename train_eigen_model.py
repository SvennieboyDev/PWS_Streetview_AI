"""
Traint een eigen geolocatiemodel voor Nederland en test het op de bevroren
testset.

Opzet ("transfer learning"):
  1. StreetCLIP blijft ongewijzigd (bevroren). Het maakt van elke foto een
     vingerafdruk: een lijst van 768 getallen (een embedding).
  2. Daarop trainen we een klein model met de Nederlandse trainingsfoto's:
       - Vakjes-model: Nederland wordt verdeeld in vakjes van X km. Een
         logistische regressie leert welk vakje bij welke vingerafdruk hoort.
           "beste vakje" = midden van het meest waarschijnlijke vakje
           "gewogen"     = gemiddelde van alle vakjes, gewogen naar kans
       - kNN: zoekt de k trainingsfoto's die het meest op de testfoto lijken
         en voorspelt het gemiddelde van hun locaties.
  3. De instellingen (vakgrootte, C, k) worden gekozen met een validatieset
     die uit de TRAININGSDATA komt. De testset wordt pas helemaal aan het
     eind één keer gebruikt, zodat de testscore eerlijk blijft.
  4. Leercurve: met de gekozen instellingen wordt het model opnieuw getraind
     op steeds meer trainingsfoto's (500, 1000, 2000, ...), elke keer 3 keer
     met een andere willekeurige selectie, en gemeten op de validatieset.
     Daalt de fout bij de meeste foto's nog, dan helpt meer data waarschijnlijk.

Uitvoer (map 'resultaten_eigen/'):
  samenvatting.csv    - zelfde maten als bij evalueer_modellen.py
  vergelijking.csv    - elk model tegen de middelpunt-baseline, met
                        betrouwbaarheidsinterval en McNemar-toets
  instellingen.csv    - scores van alle geprobeerde instellingen (validatie)
  leercurve.csv/.png  - fout op de validatieset tegen het aantal trainingsfoto's
  voorspellingen.csv  - per foto per methode
  fout_cumulatief.png, kaart_fouten.png

Benodigd:
  pip install scikit-learn   (plus wat al in requirements.txt staat)

Gebruik (in de projectmap, na mapillary_trainset.py):
  python train_eigen_model.py

Stap 1 is het zware deel: zonder grafische kaart ongeveer 0,5-1 seconde per
foto. De vingerafdrukken worden opgeslagen in de map 'embeddings/', dus dat
hoeft maar één keer, en na een onderbreking gaat het verder waar het was.
Stap 2 t/m 4 duren daarna samen zo'n 5-15 minuten.
Leercurve overslaan: python train_eigen_model.py --zonder-leercurve
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import warnings
from math import comb
from pathlib import Path

import numpy as np

import evalueer_modellen as ev
from mapillary_testset import laad_provincies

# ---------------- Instellingen ----------------
TRAIN_CSV = Path("trainset/trainset.csv")
TRAIN_MAP = Path("trainset")
EMB_MAP = Path("embeddings")
UIT = Path("resultaten_eigen")
ENCODER = "geolocal/StreetCLIP"
BATCH = 16

VAK_GROOTTES_KM = [10, 20, 40, 60, 80]  # te proberen vakgroottes
C_WAARDEN = [1.0, 10.0, 100.0]          # te proberen regularisatie (hoger = minder streng)
K_WAARDEN = [1, 5, 10, 25, 50]      # te proberen k voor kNN
VALIDATIE_DEEL = 0.2
SEED = 0
LEERCURVE_AANTALLEN = [500, 1000, 2000, 3000]  # + altijd het maximum
LEERCURVE_HERHALINGEN = 3
# ----------------------------------------------


# ---------- Stap 1: vingerafdrukken ----------

def lees_csv(pad, foto_map):
    with open(pad, newline="", encoding="utf-8") as f:
        return [{**r, "lat": float(r["lat"]), "lon": float(r["lon"])}
                for r in csv.DictReader(f) if (foto_map / r["bestand"]).exists()]


def laad_encoder():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(ENCODER)
    processor = CLIPProcessor.from_pretrained(ENCODER)
    return model, processor


class Encoder:
    """Laadt StreetCLIP pas als er echt iets te berekenen valt."""

    def __init__(self):
        self.model = None

    def __call__(self, afbeeldingen):
        import torch
        import torch.nn.functional as F
        if self.model is None:
            print("StreetCLIP laden...")
            self.apparaat = ev.kies_apparaat()
            self.model, self.processor = laad_encoder()
            self.model = self.model.to(self.apparaat).eval()
            if self.apparaat == "cuda":
                self.model = self.model.half()
        invoer = self.processor(images=afbeeldingen, return_tensors="pt")
        x = invoer["pixel_values"].to(self.apparaat)
        if self.apparaat == "cuda":
            x = x.half()
        with torch.no_grad():
            v = ev.als_tensor(self.model.get_image_features(pixel_values=x))
        return F.normalize(v.float(), dim=1).cpu().numpy().astype(np.float32)


def vingerafdrukken(naam, fotos, foto_map, encoder):
    """Geeft (matrix, fotos_met_vingerafdruk) terug. Slaat tussendoor op."""
    from PIL import Image
    EMB_MAP.mkdir(exist_ok=True)
    pad = EMB_MAP / f"{naam}.npz"
    klaar = {}
    if pad.exists():
        # 'with' sluit het bestand meteen weer; Windows weigert anders om het
        # later te vervangen ("Toegang geweigerd").
        with np.load(pad) as z:
            klaar = dict(zip(z["bestanden"].tolist(), np.array(z["vectoren"])))

    def opslaan():
        # Eerst naar een tijdelijk bestand, dan pas vervangen: zo raakt het
        # bestand nooit beschadigd als je het script halverwege afsluit.
        namen = list(klaar)
        tijdelijk = pad.with_name(pad.name + ".tmp")
        with open(tijdelijk, "wb") as f:
            np.savez(f, bestanden=np.array(namen),
                     vectoren=np.stack([klaar[n] for n in namen]) if namen
                     else np.zeros((0, 768), np.float32))
        os.replace(tijdelijk, pad)

    te_doen = [f["bestand"] for f in fotos if f["bestand"] not in klaar]
    if te_doen:
        print(f"{naam}: vingerafdrukken berekenen voor {len(te_doen)} foto's...")
    for i in range(0, len(te_doen), BATCH):
        groep, afbeeldingen = [], []
        for bestand in te_doen[i:i + BATCH]:
            try:
                with Image.open(foto_map / bestand) as img:
                    afbeeldingen.append(img.convert("RGB"))
                groep.append(bestand)
            except OSError:
                print(f"  {bestand} is kapot, overgeslagen")
        if groep:
            for bestand, v in zip(groep, encoder(afbeeldingen)):
                klaar[bestand] = v
        gedaan = min(i + BATCH, len(te_doen))
        if (i // BATCH) % 10 == 9 or gedaan == len(te_doen):
            opslaan()
            print(f"  {naam}: {gedaan}/{len(te_doen)}")

    bruikbaar = [f for f in fotos if f["bestand"] in klaar]
    return np.stack([klaar[f["bestand"]] for f in bruikbaar]), bruikbaar


# ---------- Stap 2: modellen ----------

def vakje_codes(lats, lons, km):
    dlat = km / 111.0
    dlon = km / (111.0 * math.cos(math.radians(52.2)))
    return np.array([f"{int((la - 50.0) // dlat)}_{int((lo - 3.0) // dlon)}"
                     for la, lo in zip(lats, lons)])


class VakjesModel:
    def __init__(self, km, C):
        self.km, self.C = km, C

    def fit(self, X, lats, lons):
        from sklearn.linear_model import LogisticRegression
        codes = vakje_codes(lats, lons, self.km)
        self.klassen = sorted(set(codes))
        index = {c: i for i, c in enumerate(self.klassen)}
        y = np.array([index[c] for c in codes])
        # Midden van een vakje = gemiddelde plek van de trainingsfoto's erin
        self.midden = np.array([[lats[y == i].mean(), lons[y == i].mean()]
                                for i in range(len(self.klassen))])
        self.clf = LogisticRegression(C=self.C, max_iter=500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.clf.fit(X, y)
        return self

    def voorspel(self, X):
        kans = self.clf.predict_proba(X)
        midden = self.midden[self.clf.classes_]
        beste = midden[kans.argmax(1)]
        gewogen = kans @ midden
        return beste, gewogen


def knn_voorspel(X_train, lats, lons, X, k):
    gelijkenis = X @ X_train.T  # vingerafdrukken zijn genormaliseerd: dit is cosinus
    k = min(k, len(X_train))
    buren = np.argpartition(-gelijkenis, k - 1, axis=1)[:, :k]
    return np.stack([lats[buren].mean(1), lons[buren].mean(1)], axis=1)


def mediane_fout(pred, lats, lons):
    return statistics.median(ev.afstand_km(a, b, p[0], p[1])
                             for a, b, p in zip(lats, lons, pred))


def splits_op_zoekpunt(train):
    """Validatieset op basis van zoekpunt: foto's uit hetzelfde zoekvakje
    komen samen in train óf validatie, nooit verdeeld (anders spiekt het)."""
    groepen = sorted({f.get("zoekpunt") or f["bestand"] for f in train})
    rng = random.Random(SEED)
    rng.shuffle(groepen)
    val_groepen = set(groepen[:int(len(groepen) * VALIDATIE_DEEL)])
    val = np.array([(f.get("zoekpunt") or f["bestand"]) in val_groepen for f in train])
    return ~val, val


def kies_instellingen(X, lats, lons, train):
    """Probeert alle instellingen op de validatieset en kiest de beste."""
    tr, va = splits_op_zoekpunt(train)
    print(f"\nInstellingen kiezen: {tr.sum()} foto's om te trainen, "
          f"{va.sum()} om te valideren (de testset blijft onaangeroerd).")
    rijen = []
    for km in VAK_GROOTTES_KM:
        for C in C_WAARDEN:
            m = VakjesModel(km, C).fit(X[tr], lats[tr], lons[tr])
            beste, gewogen = m.voorspel(X[va])
            rij = {"methode": "vakjes", "vak_km": km, "C": C, "k": "",
                   "val_mediaan_beste_km": round(mediane_fout(beste, lats[va], lons[va]), 1),
                   "val_mediaan_gewogen_km": round(mediane_fout(gewogen, lats[va], lons[va]), 1)}
            rijen.append(rij)
            print(f"  vakjes {km:>2} km, C={C:<5g} -> mediaan {rij['val_mediaan_beste_km']} km "
                  f"(gewogen {rij['val_mediaan_gewogen_km']} km)")
    for k in K_WAARDEN:
        pred = knn_voorspel(X[tr], lats[tr], lons[tr], X[va], k)
        rij = {"methode": "kNN", "vak_km": "", "C": "", "k": k,
               "val_mediaan_beste_km": round(mediane_fout(pred, lats[va], lons[va]), 1),
               "val_mediaan_gewogen_km": ""}
        rijen.append(rij)
        print(f"  kNN k={k:<3} -> mediaan {rij['val_mediaan_beste_km']} km")

    vakjes = [r for r in rijen if r["methode"] == "vakjes"]
    beste_vak = min(vakjes, key=lambda r: r["val_mediaan_beste_km"])
    beste_gew = min(vakjes, key=lambda r: r["val_mediaan_gewogen_km"])
    beste_knn = min((r for r in rijen if r["methode"] == "kNN"),
                    key=lambda r: r["val_mediaan_beste_km"])
    return rijen, beste_vak, beste_gew, beste_knn


# ---------- Leercurve ----------

def leercurve(X, lats, lons, train, beste_vak, beste_gew, beste_knn):
    """Traint met de gekozen instellingen op steeds meer foto's en meet de
    mediane fout op de validatieset. Geeft een lijst met rijen terug."""
    tr, va = splits_op_zoekpunt(train)
    tr_idx = np.flatnonzero(tr)
    # Aantallen vlak onder het maximum overslaan (anders twee bijna gelijke punten)
    aantallen = sorted({n for n in LEERCURVE_AANTALLEN if n <= 0.9 * len(tr_idx)} | {len(tr_idx)})
    namen = {
        "vak": f"Eigen: vakjes {beste_vak['vak_km']} km",
        "gew": f"Eigen: gewogen {beste_gew['vak_km']} km",
        "knn": f"Eigen: kNN k={beste_knn['k']}",
    }
    zelfde_model = (beste_gew["vak_km"], beste_gew["C"]) == (beste_vak["vak_km"], beste_vak["C"])
    print(f"\nLeercurve: {', '.join(map(str, aantallen))} trainingsfoto's, "
          f"{LEERCURVE_HERHALINGEN}x herhaald, gemeten op {va.sum()} validatiefoto's")
    rijen = []
    for n in aantallen:
        herhalingen = 1 if n == len(tr_idx) else LEERCURVE_HERHALINGEN
        for h in range(herhalingen):
            rng = np.random.default_rng(SEED + 100 * h + n)
            deel = rng.permutation(tr_idx)[:n]
            Xd, la, lo = X[deel], lats[deel], lons[deel]
            m = VakjesModel(beste_vak["vak_km"], beste_vak["C"]).fit(Xd, la, lo)
            beste, gewogen = m.voorspel(X[va])
            if not zelfde_model:
                _, gewogen = VakjesModel(beste_gew["vak_km"], beste_gew["C"]).fit(
                    Xd, la, lo).voorspel(X[va])
            knn = knn_voorspel(Xd, la, lo, X[va], beste_knn["k"])
            for sleutel, pred in (("vak", beste), ("gew", gewogen), ("knn", knn)):
                rijen.append({"methode": namen[sleutel], "aantal_trainingsfotos": n,
                              "herhaling": h + 1,
                              "val_mediane_fout_km": round(mediane_fout(pred, lats[va], lons[va]), 1)})
        laatste = [r for r in rijen if r["aantal_trainingsfotos"] == n]
        samen = ", ".join(f"{namen[k].replace('Eigen: ', '')}: "
                          f"{statistics.median(r['val_mediane_fout_km'] for r in laatste if r['methode'] == namen[k]):.1f} km"
                          for k in ("vak", "gew", "knn"))
        print(f"  {n:>5} foto's -> {samen}")

    # Ter vergelijking: de middelpunt-baseline op dezelfde validatiefoto's
    mid = np.array([[lats[va].mean(), lons[va].mean()]] * int(va.sum()))
    baseline = round(mediane_fout(mid, lats[va], lons[va]), 1)
    return rijen, baseline


def grafiek_leercurve(rijen, baseline, pad):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=150, facecolor=ev.OPPERVLAK)
    ev.stijl_assen(ax)
    alle_aantallen = sorted({r["aantal_trainingsfotos"] for r in rijen})
    for methode in dict.fromkeys(r["methode"] for r in rijen):
        aantallen = sorted({r["aantal_trainingsfotos"] for r in rijen if r["methode"] == methode})
        waarden = [[r["val_mediane_fout_km"] for r in rijen
                    if r["methode"] == methode and r["aantal_trainingsfotos"] == n] for n in aantallen]
        kleur = ev.KLEUREN.get(methode, ev.TEKST_2)
        ax.fill_between(aantallen, [min(w) for w in waarden], [max(w) for w in waarden],
                        color=kleur, alpha=0.15, linewidth=0)
        ax.plot(aantallen, [statistics.median(w) for w in waarden], color=kleur, lw=2,
                marker="o", markersize=5, markeredgecolor=ev.OPPERVLAK, label=methode)
    ax.axhline(baseline, color="#8a8983", lw=1.6, linestyle=(0, (5, 3)),
               label="Middelpunt (baseline)")
    ax.set_ylim(bottom=0)
    ax.set_xticks(alle_aantallen)
    ax.grid(True, color=ev.RAND, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Aantal trainingsfoto's", color=ev.TEKST_2, fontsize=10)
    ax.set_ylabel("Mediane fout op de validatieset (km)", color=ev.TEKST_2, fontsize=10)
    ax.set_title("Helpt meer data?", color=ev.TEKST, fontsize=13, loc="left",
                 pad=28, fontweight="bold")
    ax.text(0, 1.02, "Lager = beter. De lichte band toont de spreiding over "
            f"{LEERCURVE_HERHALINGEN} willekeurige selecties.", transform=ax.transAxes,
            color=ev.TEKST_2, fontsize=9, va="bottom")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
                    frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(ev.TEKST)
    fig.tight_layout()
    fig.savefig(pad, facecolor=ev.OPPERVLAK, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# ---------- Stap 3: statistiek ----------

def vergelijk_met_baseline(fotos, methoden, baseline):
    fouten = {m: np.array([ev.afstand_km(f["lat"], f["lon"], v[0], v[1])
                           for f, v in zip(fotos, vs)]) for m, vs in methoden.items()}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(fotos), (10000, len(fotos)))
    rijen = []
    for m, a in fouten.items():
        if m == baseline:
            continue
        b = fouten[baseline]
        verschil = np.median(a[idx], 1) - np.median(b[idx], 1)
        lo, hi = np.percentile(verschil, [2.5, 97.5])
        alleen_model = int(((a <= 25) & (b > 25)).sum())
        alleen_base = int(((a > 25) & (b <= 25)).sum())
        n = alleen_model + alleen_base
        p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in
                                                range(max(alleen_model, alleen_base), n + 1)) / 2 ** n)
        rijen.append({
            "methode": m, "tegen": baseline,
            "verschil_mediaan_km": round(float(np.median(a) - np.median(b)), 1),
            "interval_95_laag": round(float(lo), 1), "interval_95_hoog": round(float(hi), 1),
            "dichterbij_dan_baseline_%": round(100 * float((a < b).mean()), 1),
            "binnen_25km_model": int((a <= 25).sum()), "binnen_25km_baseline": int((b <= 25).sum()),
            "mcnemar_p_25km": float(f"{p:.2g}") if p < 0.001 else round(p, 4),
        })
    return rijen


def main():
    parser = argparse.ArgumentParser(description="Eigen geolocatiemodel trainen en testen")
    parser.add_argument("--zonder-leercurve", action="store_true", help="leercurve overslaan")
    args = parser.parse_args()

    UIT.mkdir(exist_ok=True)
    provincies = laad_provincies()

    if not TRAIN_CSV.exists():
        raise SystemExit(f"{TRAIN_CSV} niet gevonden. Draai eerst mapillary_trainset.py.")
    train = lees_csv(TRAIN_CSV, TRAIN_MAP)
    test = ev.lees_testset()
    print(f"{len(train)} trainingsfoto's, {len(test)} testfoto's.")
    if len(train) < 1000:
        print("Let op: met minder dan ~1000 trainingsfoto's leert het model weinig.")

    # Stap 1
    encoder = Encoder()
    X_train, train = vingerafdrukken("train", train, TRAIN_MAP, encoder)
    X_test, test = vingerafdrukken("test", test, ev.TESTSET_MAP, encoder)
    lats = np.array([f["lat"] for f in train])
    lons = np.array([f["lon"] for f in train])

    # Stap 2
    rijen, beste_vak, beste_gew, beste_knn = kies_instellingen(X_train, lats, lons, train)
    ev.schrijf_csv(UIT / "instellingen.csv", rijen)
    print(f"\nGekozen: vakjes {beste_vak['vak_km']} km C={beste_vak['C']:g}; "
          f"gewogen {beste_gew['vak_km']} km C={beste_gew['C']:g}; kNN k={beste_knn['k']}")

    # Leercurve (op de validatieset, de testset blijft erbuiten)
    import matplotlib
    matplotlib.use("Agg")
    kleuren = {f"Eigen: vakjes {beste_vak['vak_km']} km": "#2a78d6",
               f"Eigen: gewogen {beste_gew['vak_km']} km": "#eb6834",
               f"Eigen: kNN k={beste_knn['k']}": "#1baf7a",
               "StreetCLIP zonder training": "#eda100"}
    ev.KLEUREN.clear()
    ev.KLEUREN.update(kleuren)
    if not args.zonder_leercurve:
        lc_rijen, lc_baseline = leercurve(X_train, lats, lons, train,
                                          beste_vak, beste_gew, beste_knn)
        ev.schrijf_csv(UIT / "leercurve.csv", lc_rijen)
        grafiek_leercurve(lc_rijen, lc_baseline, UIT / "leercurve.png")

    # Opnieuw trainen op álle trainingsdata, één keer testen
    print("Eindmodellen trainen op alle trainingsdata en testen op de testset...")
    m_vak = VakjesModel(beste_vak["vak_km"], beste_vak["C"]).fit(X_train, lats, lons)
    beste, _ = m_vak.voorspel(X_test)
    if (beste_gew["vak_km"], beste_gew["C"]) == (beste_vak["vak_km"], beste_vak["C"]):
        m_gew = m_vak
    else:
        m_gew = VakjesModel(beste_gew["vak_km"], beste_gew["C"]).fit(X_train, lats, lons)
    _, gewogen = m_gew.voorspel(X_test)
    knn = knn_voorspel(X_train, lats, lons, X_test, beste_knn["k"])

    def als_methode(pred):
        return [(float(a), float(b), ev.provincie_van(float(a), float(b), provincies), "")
                for a, b in pred]

    methoden = {
        f"Eigen: vakjes {beste_vak['vak_km']} km": als_methode(beste),
        f"Eigen: gewogen {beste_gew['vak_km']} km": als_methode(gewogen),
        f"Eigen: kNN k={beste_knn['k']}": als_methode(knn),
    }

    # Ter vergelijking: StreetCLIP zonder training (uit evalueer_modellen.py)
    cache = ev.RESULTATEN / "cache_streetclip.json"
    if cache.exists():
        sc = json.loads(cache.read_text(encoding="utf-8"))
        if all(f["bestand"] in sc for f in test):
            methoden["StreetCLIP zonder training"] = [
                (sc[f["bestand"]]["lat"], sc[f["bestand"]]["lon"],
                 sc[f["bestand"]]["provincie"], sc[f["bestand"]]["gemeente"]) for f in test]

    # Dezelfde baselines als in evalueer_modellen.py
    mid_lat = statistics.mean(f["lat"] for f in test)
    mid_lon = statistics.mean(f["lon"] for f in test)
    methoden["Middelpunt (baseline)"] = [
        (mid_lat, mid_lon, ev.provincie_van(mid_lat, mid_lon, provincies), "")] * len(test)
    rng = random.Random(1)
    raster = ev.nl_raster(provincies, stap_km=1)
    methoden["Willekeurig punt (baseline)"] = []
    for _ in test:
        la, lo = rng.choice(raster)
        methoden["Willekeurig punt (baseline)"].append(
            (la, lo, ev.provincie_van(la, lo, provincies), ""))

    regels, details = ev.samenvatten(test, methoden)
    ev.schrijf_csv(UIT / "samenvatting.csv", regels)
    ev.schrijf_csv(UIT / "voorspellingen.csv", details)
    vergelijking = vergelijk_met_baseline(test, methoden, "Middelpunt (baseline)")
    ev.schrijf_csv(UIT / "vergelijking.csv", vergelijking)

    ev.print_tabel(regels)
    print("\nTegen de middelpunt-baseline (negatief verschil = beter dan de baseline):")
    for r in vergelijking:
        if r["methode"].startswith("Willekeurig"):
            continue
        print(f"  {r['methode']:<32} {r['verschil_mediaan_km']:+6.1f} km "
              f"[{r['interval_95_laag']:+.1f}, {r['interval_95_hoog']:+.1f}]  "
              f"dichterbij bij {r['dichterbij_dan_baseline_%']}%  "
              f"binnen 25 km: {r['binnen_25km_model']} vs {r['binnen_25km_baseline']} "
              f"(p {'< 0.0001' if r['mcnemar_p_25km'] < 0.0001 else '= ' + str(r['mcnemar_p_25km'])})")

    # Grafieken: eigen modellen in kleur, StreetCLIP zonder training ter vergelijking
    ev.KLEUREN.clear()
    ev.KLEUREN.update({m: k for m, k in kleuren.items() if m in methoden})
    ev.grafiek_cumulatief(regels, UIT / "fout_cumulatief.png")
    ev.kaart_fouten(test, methoden, regels, provincies, UIT / "kaart_fouten.png")
    print(f"\nKlaar. Alles staat in '{UIT}/'.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGestopt. Start opnieuw om verder te gaan.")

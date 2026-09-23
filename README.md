# Parallel Tempering – mehrere Ketten tauschen Zustände statt einen Abkühlplan zu fahren – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-parallel-tempering-demo.streamlit.app/)**

Neuntes Stück der **Trajektorien-Metaheuristiken-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Fortsetzung der [simulated-annealing-demo](../simulated-annealing-demo):
dieselbe Rundtour wie in der [hill-climbing-demo](../hill-climbing-demo), der [iterated-local-search-demo](../iterated-local-search-demo), der [variable-neighborhood-search-demo](../variable-neighborhood-search-demo), der [tabu-search-demo](../tabu-search-demo), der [grasp-demo](../grasp-demo), der [lin-kernighan-demo](../lin-kernighan-demo) und der [dynasearch-demo](../dynasearch-demo) (ein Depot, n Kundenstopps in einem 100 × 100-km-Gebiet), dieselbe untere Schranke.

**Einordnung in die Reihe:** **Parallel Tempering** (Swendsen & Wang 1986; Geyer 1991) ist eine direkte **Fortsetzung von Simulated Annealing** - ein zweiter Kind-Knoten der Wurzel Hill Climbing, kein Kind des Nachbarschafts-Zweigs (Lin-Kernighan, Dynasearch). Simulated Annealing braucht eine sorgfältig getroffene Endtemperatur; trifft man sie falsch, kommt die Kette entweder nicht zur Ruhe (zu heiß) oder verhält sich wie ein reiner Abstieg (zu kalt) - in der Schwester-Demo gemessen: 9.5 % (beste Tour) / 27.2 % (letzte Tour) bei zu heiß, 7.1 % bei zu kalt, gegen 1.4 % beim kalibrierten Standardlauf. Parallel Tempering hält stattdessen **R Ketten bei festen Temperaturen** parallel und lässt sie periodisch **Zustände tauschen** - kein Abkühlplan nötig.
```
hill-climbing-demo (Wurzel: nur bergab, bleibt im ersten Optimum stecken)        [gebaut]
  ├─ simulated-annealing-demo (nimmt Verschlechterungen an, Abkühlplan)          [gebaut]
  │     └─ parallel-tempering-demo (mehrere feste Temperaturen, Austausch)       [dieses Stück]
  ├─ iterated-local-search-demo (stört ein gutes Optimum mit fester Störstärke)  [gebaut]
  │     └─ variable-neighborhood-search-demo (Störstärke eskaliert + Reset)     [gebaut]
  ├─ tabu-search-demo (immer der beste Zug, Gedächtnis gegen Rückwege)          [gebaut]
  ├─ grasp-demo (randomisierte Konstruktion, viele Neuanfänge)                  [gebaut]
  └─ Nachbarschafts-Zweig
        ├─ lin-kernighan-demo (variable Tiefe statt fixer 2-opt-Nachbarschaft)  [gebaut]
        └─ dynasearch-demo (viele unabhängige Züge auf einmal statt einer)      [gebaut]
              └─ VRP-Nachbarschaften (inter-route-Züge)                        [nicht gebaut]
```

Ergebnis in Kürze: **löst Parallel Tempering das Tuning-Problem von Simulated Annealing? Teilweise.** Eine ENGE, gut platzierte Temperaturleiter (0.1-0.3 mittlere Kantenlängen) schlägt bei 60 Stopps und dem Standardbudget (200 Tausend) kalibriertes Simulated Annealing knapp (1.15 % gegen 1.42 % über der Schranke) - aber eine schlecht gewählte Leiter kostet spürbar (3-6x schlechter als kalibriert): Parallel Tempering ist **NICHT tuningfrei**. Es ist aber deutlich **robuster** als Simulated Annealing gegen eine ungefähr-statt-exakt richtige Temperaturwahl - selbst eine Leiter, die genau SAs eigene Fehlkalibrierungen abdeckt (0.02 bis 0.5), bleibt mit 3.22 % weit vor SAs eigenen Ausfällen (zu kalt 8.66 %, zu heiß 14.96 % / 31.57 %). Der schärfste ehrliche Befund: Parallel Tempering teilt das Budget auf R Ketten auf - bei **kleinem Budget** (10 Tausend: 32.05 % gegen 9.01 %) oder **großen Instanzen** (200 Stopps, festes Budget: 23.55 % gegen 7.87 %) bekommt jede Kette zu wenig, um überhaupt anzukommen. Dafür **wächst** der Vorsprung mit dem Budget statt zu schrumpfen (2 Millionen: 0.49 % gegen 0.73 %) - ein Kontrast zu den meisten anderen Stücken dieser Linie.

| Frage | Ergebnis (60 gleichverteilte Stopps, kalibrierte Leiter R=5/0.1-0.3/Tausch alle 300, 200 Tausend Vorschläge, sofern nicht anders angegeben; Mittel über 5 feste Instanzen, Seeds 100000–100004, mit je 3 Ketten-Seeds; Abstand = Prozent über der 1-Baum-Schranke) |
|---|---|
| Standardfall | ✅ Parallel Tempering **1.15 %** über der Schranke gegen **1.42 %** für kalibriertes Simulated Annealing (gleiches Budget) - knapper Sieg |
| **Budget-Sweep (PT gegen SA-getunt)** | ⚠️→✅ 10T-100T: PT verliert deutlich (10T: **32.05/9.01 %**). 200T-2M: PT gewinnt, mit WACHSENDEM Vorsprung (500T: 0.77/1.27, 1M: 0.62/0.94, 2M: **0.49/0.73 %**) |
| **Skalierung bei festem Budget (200 Tausend)** | ⚠️ n=20/40/60: nah dran, nicht monoton (0.05/0.05, 0.95/0.79, 1.15/1.42). AB n=100: PT verliert deutlich (6.20/3.08, 12.95/6.54, **23.55/7.87 %** bei n=200) |
| **Leiterbreite (R=5, kalibriertes Tausch-Intervall)** | ⚠️ eng+gut platziert (0.1-0.3) **1.15 %**; SA-Extreme abgedeckt (0.02-0.5) 3.22 %; eng zu heiß (0.4-0.6) 6.62 %; eng zu kalt (0.02-0.1) 4.48 % - Feinabstimmung nötig, aber robuster als SA |
| **Replikat-Zahl R (0.1-0.3, Tausch-Intervall 300)** | ⚠️ R=2/3/5/7/10: 2.24/2.19/**1.15**/1.78/2.32 % - nicht-monotones Optimum bei R=5 |
| **Tausch-Intervall (0.1-0.3, R=5)** | ⚠️ 10/30/100/300/1000/3000: 2.05/2.03/1.41/**1.15**/1.64/2.19 % - ebenfalls nicht-monoton, Optimum bei 300 |
| **Austausch an/aus (kalibrierte Leiter)** | ✅ an **1.15 %** gegen aus 2.40 % - der Austausch selbst bringt den Vorteil (geprüft über jede der 5 Sweep-Instanzen einzeln) |

## Was die Demo zeigt

1. **Parallel Tempering in Aktion** (Schritt-Slider): **Instanz** → **Leiter im Austausch** (die PT-eigene Leiter-Heatmap: x = Runde, y = Kette nach Temperatur sortiert, Farbe = Abstand zur Schranke - zeigt sichtbar, wie gute Touren von heißen zu kalten Ketten durchsickern, dazu die Tausch-Annahmequote je Nachbarpaar) → **Ergebnis** (beste Tour über alle Ketten neben kalibriertem Simulated Annealing bei gleichem Budget).
2. **Was die Suche gefunden hat:** Abstand zur Schranke für Parallel Tempering, kalibriertes Simulated Annealing und Hill Climbing mit Neustarts; Urteil (`beats_sa` → `comparable` → `sa_wins`), Detailtabellen.
3. **📐 Sweeps** über Budget, Anzahl Replikate, Tausch-Intervall, Stopps und Gruppen (feste Instanzen ab 100000, drei Ketten je Instanz).
4. **🔬 Experimente auf Abruf:** Budget von 10 Tausend bis 2 Millionen (der zentrale, wachsende Vorsprung); Streuung über 20 Ketten; Skalierung von 20 bis 200 Stopps bei festem und wachsendem Budget.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (Budget-Teilung auf R Ketten, ungefähr richtige Leiter, nur Metropolis-Annahme).

Regler: Stopps (10–200), Anteil der Stopps in Gruppen, **Anzahl Replikate R** (2–10), **T_min/T_max** (die Temperaturleiter, Vielfache der mittleren Kantenlänge einer guten Tour, wie SAs T0/T_end), **Tausch-Intervall** (Vorschläge je Kette zwischen zwei Tausch-Versuchen), **Austausch an/aus** (Kontroll-Umschalter), Budget (10 Tausend bis 2 Millionen bewertete Nachbarn), Seed der Instanz (+ 🎲), Seed der Kette (+ 🎲, steuert nur die gemeinsame Startlösung - der Kern ist sonst deterministisch bis auf den Metropolis-Zufall selbst).
Nur Metropolis-Annahme (die Tausch-Formel setzt eine wohldefinierte Gleichgewichtsverteilung je Kette voraus - SAs andere drei Regeln haben keine, siehe Grenzen). Kein Nachbarschafts- oder Abkühlplan-Regler (Simulated Annealing hat diese Fragen bereits geklärt; hier bewusst nur 2-opt und feste statt sich ändernder Temperaturen, um genau den Unterschied "Plan gegen Leiter+Austausch" zu isolieren).

## Messwerte der Presets (Instanz-Seed 35, Ketten-Seed 0; sie prüfen sich mit Urteil-Bändern selbst)

| Preset | Urteil (Band über Instanzen × Ketten) |
|---|---|
| Standardfall (Voreinstellung) | beats_sa / comparable / sa_wins |
| Ohne Tausch (Kontrolle) | beats_sa / comparable / sa_wins |
| Weite Leiter (deckt SA-Extreme ab) | beats_sa / comparable / sa_wins |
| Kleines Budget (10 Tausend) | sa_wins |
| Großes Budget (1 Million) | beats_sa / comparable / sa_wins |
| Große Instanz (200 Stopps) | sa_wins |

"Kleines Budget" und "Große Instanz" sind über JEDE getestete Instanz/Kette ein durchgehender Negativbefund - der klarste, am wenigsten mehrdeutige Teil des ganzen Stücks. Bei der einzelnen Standardinstanz (Seed 35) fällt der Austausch-Effekt schwächer aus als im Sweep-Mittel (dort ist er über jede der 5 Instanzen einzeln bestätigt, siehe Verifikation) - ein Hinweis, dass gerade diese eine Instanz für den Austausch-Vergleich nicht repräsentativ ist; die Mittelwerte sind die belastbaren Zahlen.

## Modell und Verfahren

- **Instanz, Nachbarschaften, Abstieg, Schranke** (`pt_scenario.py`, `pt_tour.py`): wortgleiche Kopien aus der [hill-climbing-demo](../hill-climbing-demo).
- **Metropolis-Kern bei fester Temperatur** (`pt_algorithm._chain_segment`): eine PORTIERTE Teilmenge von `simulated-annealing-demo/sa_algorithm.py`s `anneal()`-Funktion (nur der Metropolis-Zweig, feste Temperatur statt Plan, nur 2-opt) - kein eigenständig hergeleitetes Risiko im Kern-Innenloop. Verfolgt laufend die BESTE innerhalb des Segments erreichte Tour mit, nicht nur den End-Zustand (ein echter, gefundener Fehler - siehe Verifikation).
- **Temperaturleiter** (`pt_algorithm.temperature_ladder`): R Temperaturen, geometrisch gestuft von T_min bis T_max - dieselbe Formel wie SAs geometrischer Abkühlplan, hier aber als FESTE Leiter statt eines zeitlichen Plans; dieselbe Funktion erzeugt auch die SA-Vergleichsgröße (`pt_algorithm.sa_baseline`, ebenfalls auf `_chain_segment` aufgebaut).
- **Der Tausch-Zug** (`pt_algorithm.swap_probability`, `parallel_tempering`): alle `Tausch-Intervall` Vorschläge je Kette wird ein Tausch zwischen benachbarten Ketten im Schachbrett-Schema vorgeschlagen, angenommen mit der Formel, die die gemeinsame Boltzmann-Verteilung über alle Ketten erhält (siehe Mathe-Expander in der App). Ein Tausch zählt NICHT als bewerteter Nachbar (dieselbe Konvention wie ein ILS-Kick).
- **SA-Vergleichsgröße** (`pt_algorithm.sa_baseline`): kalibrierter Standardlauf der Simulated-Annealing-Demo (T0=0.5, T_end=0.1, geometrisch, 100 Stufen), hier mit demselben Metropolis-Baustein nachgebaut und gegen die echte `sa_algorithm.anneal()` kreuzgeprüft (siehe Verifikation).
- **Auswertung** (`pt_evaluation.py`): Kennzahlen, Urteil, Sweeps über feste Instanzen × Ketten, Streuung, Skalierung.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutung: "eine breite Leiter braucht keine Feinabstimmung"** – **widerlegt**. Eine breite Leiter, die SAs eigene Fehlkalibrierungs-Extreme abdeckt (0.02-0.5), liegt bei 3.22 % - klar schlechter als eine enge, gut platzierte Leiter (1.15 %) und sogar schlechter als zwei enge, aber leicht daneben liegende Leitern (0.4-0.6: 6.62 %; 0.02-0.1: 4.48 %, die schlechteste getestete). Parallel Tempering braucht also eine ungefähr richtig ZENTRIERTE Leiter, nicht nur eine breite. Trotzdem bleibt selbst die schlechteste getestete Leiter (6.62 %) weit vor Simulated Annealings eigenen Fehlkalibrierungen (zu kalt 8.66 %, zu heiß 14.96 % beste / 31.57 % letzte Tour) - **robuster, aber nicht tuningfrei** ist die ehrliche Formulierung.
- **Das Budget wird durch R geteilt - ein strukturelles Handicap bei kleinem Budget oder großen Instanzen.** Bei 10 Tausend Vorschlägen (2 Tausend je Kette bei R=5) kommt jede Kette kaum vom Fleck: 32.05 % gegen 9.01 % für SA (das keinen Split braucht). Derselbe Effekt bei festem Budget und wachsender Instanz: ab 100 Stopps reicht das geteilte Budget je Kette nicht mehr (6.20 % gegen 3.08 % bei n=100, 23.55 % gegen 7.87 % bei n=200) - ein ehrlicher, durchgehender Negativbefund, der sich auch bei wachsendem Budget (5000 · Stopps) nur abschwächt, nicht auflöst (7.20 % gegen 3.78 % bei n=200).
- **Dafür wächst der Vorsprung mit dem Budget, statt zu schrumpfen** - ein Kontrast zu GRASP, Lin-Kernighan und den meisten anderen Stücken dieser Linie, deren Vorteil bei sehr großem Budget verpufft oder sich umkehrt. Bei 60 Stopps gewinnt Parallel Tempering ab etwa 150-200 Tausend Vorschlägen und der Abstand zu SA WÄCHST danach (200T: 0.27 Punkte Vorsprung, 2M: 0.24 Punkte - absolut ähnlich, aber SAs eigener Wert schrumpft schneller, sodass der RELATIVE Vorsprung von PT wächst).
- **Nur Metropolis.** Die Tausch-Formel setzt eine wohldefinierte Boltzmann-Gleichgewichtsverteilung je Kette voraus - Threshold Accepting, Great Deluge und Late Acceptance Hill Climbing (die anderen drei Regeln der Simulated-Annealing-Demo) haben keine bekannte Gleichgewichtsverteilung, die Formel wäre für sie nicht herleitbar. Bewusst nicht umgesetzt, keine willkürliche Einschränkung.
- **Ein echter Implementierungsfehler unterwegs gefangen**: eine frühe Fassung von `_chain_segment` gab nur die Tour am ENDE des Segments zurück, nicht die BESTE währenddessen erreichte - Metropolis nimmt auch Verschlechterungen an, die Länge schwankt also innerhalb eines Segments, und ein guter Zwischenstand kann am Ende des Segments schon wieder verloren sein. Aufgefallen beim Kreuzvergleich von `sa_baseline` gegen die echte `sa_algorithm.anneal()`: bei "zu heiß" (viele angenommene Verschlechterungen, viel Schwankung) wich der Mittelwert über 40 Seeds um ~4 Prozentpunkte ab (deutlich mehr als der Stichprobenfehler erklären konnte) - beim kalibrierten, ruhigeren Lauf war die Abweichung klein genug, um beinahe unbemerkt zu bleiben. Behoben, indem `_chain_segment` die beste Tour laufend mitverfolgt und zurückgibt; seither reproduziert `sa_baseline` die echte `anneal()`-Funktion im Mittel innerhalb der Stichprobenschwankung (siehe Verifikation).
- **Synthetische Instanzen:** euklidisch, gleichverteilt oder in fünf Gruppen, ein Fahrzeug, keine Kapazitäten oder Zeitfenster. Zeiten hängen vom Rechner und der Python-Version ab (die Tests prüfen nur Größenordnungen).

## Verifikation

- **Die für Parallel Tempering eigentliche, neue Korrektheitseigenschaft:** der Tausch erhält die GEMEINSAME Boltzmann-Gleichgewichtsverteilung - jede Kette bleibt bei ihrer EIGENEN Temperatur im Gleichgewicht, auch während sie ständig mit anderen tauscht. Geprüft auf einer vollständig aufzählbaren 6-Knoten-Instanz (60 Touren, alle Permutationen bekannt): zwei gekoppelte Ketten laufen über 40000 Runden, die empirische Marginalverteilung jeder Kette wird gegen ihre theoretische Boltzmann-Verteilung verglichen (Total-Variation-Abstand < 0.06, gemessen ~0.01-0.02) - dieselbe Technik wie der Boltzmann-Test der Simulated-Annealing-Demo, hier auf zwei GETAUSCHTE Ketten erweitert. Eine absichtlich kaputte Tausch-Formel (immer annehmen) wurde probeweise gegengeprüft und verschlechtert den Total-Variation-Abstand auf über 0.20 - der Test hat also echte Trennschärfe.
- **`_chain_segment` als vertrauter Baustein statt Neuherleitung**: sowohl `parallel_tempering` als auch `sa_baseline` bauen auf demselben, unabhängig getesteten Metropolis-Kern auf. Regressionen: bei genau einem Segment je Kette (Tausch-Intervall ≥ Budget) reproduziert `parallel_tempering` bytegleich einen direkten `_chain_segment`-Aufruf (R=1 und mehrere unabhängige Ketten einzeln geprüft); bei ausgeschaltetem Austausch verhält sich Replikat 0 IDENTISCH, unabhängig davon, wie viele andere Replikate daneben laufen (bei gleichem eigenem Budget/Tausch-Intervall/Seed) - die eigentliche "R unabhängige Ketten"-Eigenschaft, unabhängig von internen Chunking-Details geprüft (siehe Hinweis im Modul-Docstring von `pt_algorithm.py`).
- **`sa_baseline` gegen die echte `sa_algorithm.anneal()` kreuzgeprüft** (Skript, nicht Teil der Test-Suite, da es die Schwester-Demo importiert): Mittelwerte über 30-40 Seeds stimmen sowohl im kalibrierten als auch im "zu heiß"-Regime innerhalb der Stichprobenschwankung überein - der Fund und die Behebung des Best-Tracking-Fehlers oben.
- Übernommener Kern: 2-opt gegen Brute-Force, Abstieg strikt monoton und im lokalen Optimum, Bewertungsbudget, 1-Baum-Schranke; Instanz gegen eingefrorene Werte (aus der Hill-Climbing-Demo).
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Budget-, Skalierungs-, Leiterbreite-, Replikat- und Tausch-Intervall-Aussagen; jeweils Mittel über die festen Sweep-Instanzen × Ketten; positive **und** negative Aussagen; Rechenzeiten nur als Größenordnung), über dieselben Auswertungsfunktionen wie die App selbst (`ev.run_config`/`ev.sweep`), NIE über ein Ad-hoc-Skript mit abweichender Zufalls-Bindung (die Lehre aus der [lin-kernighan-demo](../lin-kernighan-demo) dieser Linie); alle 6 Presets über mehrere Instanzen und Ketten in Urteil-Bändern; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt, Austausch-aus-Ansicht, Würfel-Knöpfe, Permalink-Grenzen, Extremwerte, Experimente auf Abruf, Footer).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Budget, Streuung, Skalierung), 🚧 Grenzen, Mathe |
| `pt_algorithm.py` | Der Metropolis-Kern (`_chain_segment`), die Temperaturleiter, der Tausch-Zug (`parallel_tempering`), die SA-Vergleichsgröße (`sa_baseline`) |
| `pt_tour.py` | Nachbarschaften, Abstieg (mit Bewertungsbudget), Kreuzungen, 1-Baum-Schranke (aus der Hill-Climbing-Demo) |
| `pt_scenario.py`, `pt_constants.py` | Instanzen; Konstanten, Presets |
| `pt_evaluation.py` | Analyse, Urteil, Sweeps, Streuung, Skalierung |
| `pt_presets.py`, `pt_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt, inkl. der Leiter-Heatmap) |
| `tests/` | Übernommener Kern, Parallel-Tempering-Korrektheit (Boltzmann-Kreuzprüfung, Regressionen), Szenario und Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).

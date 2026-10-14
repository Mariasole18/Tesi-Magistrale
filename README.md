# Cambiamento climatico e flussi migratori: un approccio di machine learning applicato al modello gravitazionale
Repository dei codici a supporto della tesi di Laurea Magistrale in **Data Science per le Strategie Aziendali** (Università della Calabria).
Il lavoro analizza i **flussi migratori internazionali** verso i paesi OECD attraverso un **modello gravitazionale esteso**, stimato con **Poisson Pseudo Maximum Likelihood (PPML)**. Alla metodologia tradizionale vengono affiancate tecniche di *ensemble learning*, *statistical learning* e di interpretabilità (Random Forest, Gradient Boosting, LASSO, Elastic Net, SHAP Values) con finalità selettiva, esplorativa e interpretativa, oltre a un'analisi di robustezza basata sull'introduzione della misura differenziale della componente istituzionale. L'obiettivo è dunque di tipo interpretativo, teso a individuare le componenti maggiormente rilevanti nell'analisi del fenomeno oggetto di studio. In più si persegue lo scopo di testare l'interazione tra la componente climatica e quella istituzionale, per verificare l'ipotesi secondo cui, in un contesto simile, la componente istituzionale agisca come fattore di mitigazione dell'impatto della componente climatica sui flussi.

---

## Struttura del repository

| File | Ambiente | Descrizione |
|------|----------|-------------|
| `Tesi - Costruzione Dataset.ipynb` | Python / Colab | Raccolta, integrazione e pulizia di tutte le fonti; produzione del dataset finale e dei dataset intermedi. |
| `Tesi - Metodologia - Analisi Esplorativa.ipynb` | Python / Colab | Analisi esplorativa (variabile dipendente, analisi temporale e geografica, statistiche descrittive, correlazioni, indici istituzionali sintetici, VIF). Produce il dataset post-VIF. |
| `Tesi - Metodologia - Applicazione di Tecniche per Esplorare, Selezionare e Interpretare le Variabili.ipynb` | Python / Colab | Random Forest, Gradient Boosting, LASSO, Elastic Net (con residualizzazione FWL) e SHAP Values (Summary Plot e Dependence Plot). |
| `Tesi - Metodologia - Fase di Stima Principale.do` | Stata | Stima del modello PPML con effetti fissi ad alta dimensionalità (`ppmlhdfe`), confronto tra variabili climatiche, modelli annidati e termine di interazione clima–istituzioni. |
| `Tesi - Metodologia - Analisi di Robustezza - Nuovo Dataset.ipynb` | Python / Colab | Costruzione del dataset per la robustezza: indice di corruzione (V-Dem), con relativa misura differenziale origine–destinazione. |
| `Tesi - Metodologia - Analisi di Robustezza.do` | Stata | Ri-stima PPML sulla misura differenziale istituzionale. | Ri-stima PPML mutando, due volte, la composizione dei flussi migratori. |

---

## Pipeline e ordine di esecuzione

```
1. Costruzione Dataset            →  dataset_tesi_definitivo.csv
2. Analisi Esplorativa            →  dataset_post_vif_finale.csv
3. Applicazione delle Tecniche    →  (RF, Boosting, LASSO, EN, SHAP)
4. Fase di Stima Principale       →  modello PPML
5. Robustezza – Nuovo Dataset     →  dataset_robustezza_differenziale_istituzioni.csv
6. Robustezza – Stima             →  modello PPML (misura differenziale, due composizioni differenti dei flussi migratori)
```

I passi 1–2 vanno eseguiti in sequenza perché ogni file consuma l'output del precedente. I passi 3 e 4 utilizzano entrambi `dataset_post_vif_finale.csv` (con la specifica che il passo 3, nella prima fase si serve anche di `dataset_tesi_definitivo.csv`). I passi 5–6 richiedono a monte il dataset post-VIF, sebbene il passo 6 utilizzi specificatamente `dataset_robustezza_differenziale_istituzioni.csv` .

---

## Dati

> **Nota sui percorsi.** I notebook sono stati sviluppati su Google Colab e utilizzano percorsi del tipo `/content/drive/MyDrive/...`. Per riprodurre l'analisi è necessario adattare i percorsi alla propria struttura di cartelle (es. `Dati Grezzi/` e `Dati Puliti/`).

### Fonti grezze (`Dati Grezzi/`)

| File / Fonte | Provenienza |
|--------------|-------------|
| `OECD_international_migration_flows.csv` | OECD – inflows (variabile dipendente) |
| `Stock Connazionali.csv` | OECD – stock di migranti |
| `gravity_base.csv` | CEPII – Gravity database |
| `geo_cepii.xls` | CEPII – GeoDist |
| *World Bank* (via API `pandas-datareader`) | indicatori macroeconomici |
| `qog_std_ts_jan26.csv` | QoG Standard / Worldwide Governance Indicators |
| `vulnerability.csv` | ND-GAIN (University of Notre Dame) |
| `public_emdat_custom_request_*.xlsx` | EM-DAT (CRED) |
| `gistemp1200_GHCNv4_ERSSTv5.nc.gz` | NASA GISTEMP |
| `average-exposure-pm25-pollution.csv` | Our World in Data (PM2.5) |
| `forest-area-as-share-of-land-area.csv` | Our World in Data / FAO |
| V-Dem Public Corruption Index (via QoG) | *(robustezza)* |

Alcune fonti (World Bank) sono interrogate direttamente e non salvate come file grezzo.

### Dataset intermedi (`Dati Puliti/`) e riproducibilità

L'analisi è organizzata come una pipeline sequenziale in cui gli output di ciascuna fase costituiscono gli input delle fasi successive. Per riprodurre l'analisi è pertanto necessario:
1. disporre delle fonti dati indicate nella sezione **Dati**;
2. adattare i percorsi dei file alla propria struttura locale;
3. eseguire i notebook nell'ordine indicato nella pipeline;
4. disporre delle librerie Python e dei pacchetti Stata indicati nella
   sezione **Ambiente computazionale**.

Inoltre, sempre per garantire la **riproducibilità** senza dover rieseguire l'intera catena di integrazione, il flusso produce i seguenti dataset intermedi (output progressivi del notebook di costruzione):

```
oecd_flussi_puliti
  └─ oecd_cepii_unito.csv                     (+ CEPII gravity)
     └─ oecd_cepii_geografico.csv             (+ landlocked)
        └─ oecd_cepii_variabili_macroeconomiche.csv   (+ World Bank)
           └─ oecd_cepii_aggiunta_comcur.csv        (+ common currency)
              └─ oecd_cepii_connazionali.csv         (+ stock connazionali)
                 └─ oecd_cepii_wb_qog.csv            (+ Schengen, WGI)
                    └─ oecd_aggiunta_ndgain_emdat.csv   (+ ND-GAIN, EM-DAT)
                       └─ oecd_aggiunta_nasa.csv        (+ GISTEMP)
                          └─ oecd_aggiunta_foreste_PM2_5.csv (+ PM2.5, foreste)
                             └─ dataset_tesi_definitivo.csv   (pulizia finale NaN)
```

Dataset a valle dell'analisi esplorativa e della robustezza:

- `dataset_post_vif_finale.csv` — dataset di lavoro usato dalle tecniche ML e dalla stima PPML (indici istituzionali sintetici, rimozione variabili ridondanti via correlazione/VIF).
- `dataset_robustezza_differenziale_istituzioni.csv` — dataset con la misura differenziale istituzionale (V-Dem) per l'analisi di robustezza.

Il dataset finale, utilizzato ai fini della stima principale, comprende **63.714 osservazioni** relative a 158 paesi di origine, 31 paesi di destinazione e al periodo 1996–2020 (panel lievemente sbilanciato, annualità 1997, 1999 e 2001 assenti).

---
Inoltre, ai fini della replicabilità, per l'esecuzione completa delle ultime due verifiche di robustezza è necessario disporre anche dei file in cui i flussi migratori sono costituiti utilizzando categorie di reddito della World Bank differenti da quelle del modello benchmark. Difatti, nella prima verifica di robustezza il campione comprende i flussi originati dai paesi appartenenti alla categoria low income (LIC) e diretti verso paesi appartenenti alle categorie lower middle income, upper middle income e high income (LMC, UMC e HIC). Nella seconda, l'analisi viene ulteriormente circoscritta ai flussi provenienti dai paesi LIC, LMC e UMC mantenendo come destinazioni i paesi appartenenti alle rimanente categoria di reddito (HIC). Basterà andare nel file dedito alla creazione dei dati e sostituire gli opportuni acronimi in fase di gestione della variabile dipendente, per poi eseguire tutte le celle rimanenti del file corrente e del file di analisi esplorativa (così da disporre del df ridotto). 

## Ambiente computazionale

**Python** (notebook): `pandas`, `numpy`, `pandas-datareader` nello specifico modulo `wb`, `pycountry`, `pycountry_convert`, `xarray`, `netCDF4`, `h5netcdf`, `subprocess`, `sys`, `gzip`, `shutil`, `os`, `regionmask`, `cartopy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `StandardScaler`, `GridSearchCV`, `shap`, `PartialDependenceDisplay`, `KMeans`.

**Stata** (stima): `ppmlhdfe`, `reghdfe`, `ftools`, `kountry`.

```stata
ssc install ppmlhdfe, replace
ssc install reghdfe, replace
ssc install ftools
ssc install kountry
```

---

## Licenza

Il codice e i materiali computazionali sviluppati nell'ambito del progetto sono distribuiti secondo i termini della **MIT License**. I dati provenienti da fonti esterne rimangono soggetti alle condizioni d'uso e alle licenze stabilite dai rispettivi fornitori. Consultare il file `LICENSE` per maggiori informazioni.

---

## Autrice

**Mariasole Greco**
Relatrici: Prof.ssa Mariarosaria Agostino, Prof.ssa Sabrina Giordano
Anno Accademico 2025/2026

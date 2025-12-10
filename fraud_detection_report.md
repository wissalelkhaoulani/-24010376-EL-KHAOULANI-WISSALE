# Cours de science de données 
# 24010376
# CAC2
# EL KHAOULANI WISSALE
# Ecole national de commerce et de gestion
<img src="photo.jpg" style="height:150px;margin-right:100px"/> 



# RAPPORT PROFESSIONNEL D'ANALYSE
# Projet : Système de Détection de Fraude Financière
# Dataset : Fraud Guard Synthetic 2025

---

**Auteur** : Data Science Team  
**Date** : Décembre 2025  
**Type de Projet** : Classification Binaire Supervisée  
**Criticité** : Haute (Enjeux financiers et réglementaires)

---

## 📋 SOMMAIRE EXÉCUTIF

Ce rapport détaille l'implémentation d'un système de détection de fraude basé sur l'apprentissage automatique, appliqué au dataset **Fraud Guard Synthetic 2025**. Le projet suit rigoureusement les 7 phases du cycle de vie standard de la Data Science, depuis l'analyse métier jusqu'à l'audit de performance.

**Résultats clés attendus** :
- Identification automatique des transactions frauduleuses
- Minimisation des faux négatifs (fraudes non détectées)
- Optimisation du ROC AUC Score (>0.90 visé)

---

## 1️⃣ LE CONTEXTE MÉTIER ET LA MISSION

### 🎯 **Le Problème (Business Case)**

Dans le secteur bancaire et financier, la fraude représente un défi critique avec des conséquences multiples :

- **Pertes financières directes** : Milliards de dollars perdus annuellement
- **Réputation** : Érosion de la confiance des clients
- **Conformité réglementaire** : Obligations légales strictes (PSD2, GDPR)
- **Impact psychologique** : Stress et perte de confiance des victimes

**Objectif stratégique** : Développer un "Assistant IA" en temps réel capable d'analyser les transactions et de signaler automatiquement les comportements suspects.

---

### ⚖️ **L'Enjeu Critique : Matrice des Coûts d'Erreur Asymétrique**

Contrairement au diagnostic médical, ici la matrice des coûts est inversée mais tout aussi critique :

| Type d'Erreur | Impact | Coût Métier | Priorité |
|---------------|--------|-------------|----------|
| **Faux Positif (FP)** | Bloquer une transaction légitime | ⚠️ **Moyen** : Frustration client, appels support, perte de ventes | Modérée |
| **Faux Négatif (FN)** | Laisser passer une fraude | 🔴 **CRITIQUE** : Perte financière directe, responsabilité légale | **MAXIMALE** |

**⚠️ RÈGLE D'OR** : Le système doit **maximiser le Recall (Sensibilité)** pour capturer le maximum de fraudes, quitte à générer quelques fausses alertes qui seront validées manuellement par l'équipe anti-fraude.

**Contrainte secondaire** : Maintenir une **Precision raisonnable** (>70%) pour éviter de saturer les équipes humaines avec des alertes inutiles.

---

### 📊 **Les Données (L'Input)**

#### **Source** : Fraud Guard Synthetic 2025 (Kaggle)
- **Nature** : Dataset synthétique généré pour simuler des transactions financières réalistes
- **Avantage** : Conformité GDPR (pas de données personnelles réelles)
- **Structure attendue** :

```
Variables typiques dans un dataset de fraude financière :
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 Features Temporelles :
   - timestamp / step : Moment de la transaction
   - hour / day_of_week : Patterns temporels

📌 Features Transactionnelles :
   - amount : Montant de la transaction
   - type : Type (PAYMENT, TRANSFER, CASH_OUT, etc.)
   - oldbalanceOrg / newbalanceOrig : Soldes émetteur
   - oldbalanceDest / newbalanceDest : Soldes destinataire

📌 Features Identifiants :
   - nameOrig : ID client émetteur
   - nameDest : ID client destinataire

📌 Target (y) :
   - is_fraud / isFraud : Variable binaire (0 = Légitime, 1 = Fraude)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Caractéristique critique** : **Déséquilibre extrême des classes**
- Fraudes réelles : ~0.1% à 3% des transactions (classe minoritaire)
- Transactions légitimes : ~97-99.9% (classe majoritaire)

---

## 2️⃣ LE CODE PYTHON (LABORATOIRE)

### 🧪 **Architecture du Script**

Le script suit un pattern industriel modulaire en 8 phases :

```python
# PHASE 1 : Acquisition & Téléchargement (KaggleHub)
# PHASE 2 : Exploration Initiale (Info, Stats, NaN)
# PHASE 3 : Data Wrangling (Nettoyage, Imputation, Encodage)
# PHASE 4 : EDA Avancée (Visualisations, Corrélations)
# PHASE 5 : Feature Engineering (Création de variables dérivées)
# PHASE 6 : Protocole Expérimental (Train/Test Split Stratifié)
# PHASE 7 : Intelligence Artificielle (Random Forest + Class Balancing)
# PHASE 8 : Audit de Performance (Métriques, Courbe ROC, Feature Importance)
```

### 📦 **Stack Technologique**

```python
import numpy as np                    # Calcul matriciel
import pandas as pd                   # Manipulation de données tabulaires
import matplotlib.pyplot as plt       # Visualisation statique
import seaborn as sns                 # Visualisation statistique avancée
import kagglehub                      # Interface Kaggle

# Scikit-Learn : La référence ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    confusion_matrix, 
    roc_auc_score,      # Métrique clé pour déséquilibre
    roc_curve           # Courbe performance
)
```

---

## 3️⃣ ANALYSE APPROFONDIE : NETTOYAGE (DATA WRANGLING)

### 🔧 **Le Problème Mathématique du "Vide"**

Les valeurs manquantes (NaN, NULL, None) sont toxiques pour les algorithmes :

1. **Algèbre linéaire** : Une seule valeur manquante dans une matrice rend impossible le calcul de distances (euclidienne, Manhattan)
2. **Arbres de décision** : Peuvent gérer les NaN nativement, mais la performance est sous-optimale
3. **Réseaux de neurones** : Incompatibilité totale avec les NaN

**Diagnostic** :
```python
# Identification des colonnes problématiques
missing_summary = df.isnull().sum()
missing_pct = (missing_summary / len(df)) * 100

# Règle de décision :
# - Si < 5% manquant : Imputation
# - Si 5-30% manquant : Imputation + flag binaire "était_manquant"
# - Si > 30% manquant : Supprimer la colonne
```

---

### 🛠️ **La Mécanique de l'Imputation**

#### **Stratégie pour Variables Numériques** :

```python
imputer = SimpleImputer(strategy='mean')  # ou 'median' si outliers

# Étape 1 : Apprentissage (fit)
# L'imputer scanne la colonne "amount" sur le Train Set
# Calcul : μ = 2,347.89 € (moyenne)
# Stockage en mémoire : imputer.statistics_

# Étape 2 : Transformation (transform)
# Repasse sur les données et remplace NaN par μ
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)  # Utilise μ du train !
```

**Alternatives selon le contexte** :
- `strategy='median'` : Robuste aux outliers (montants extrêmes)
- `strategy='most_frequent'` : Pour variables catégorielles
- KNN Imputer : Utilise les K voisins les plus proches (plus coûteux)

---

### ⚠️ **Le Coin de l'Expert : Data Leakage (Fuite de Données)**

**ERREUR FATALE À ÉVITER** :

```python
# ❌ MAUVAIS : Imputation AVANT séparation Train/Test
X_imputed = imputer.fit_transform(X)  # Utilise TOUTES les données
X_train, X_test = train_test_split(X_imputed, ...)

# Pourquoi c'est grave ?
# La moyenne calculée inclut des informations du futur (test set)
# Le modèle aura "vu" indirectement les données de test
# Les performances seront surestimées de 2-5%
```

**✅ BONNE PRATIQUE INDUSTRIELLE** :

```python
# 1. Séparer d'abord
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)

# 2. Apprendre l'imputation sur le train uniquement
imputer.fit(X_train)  

# 3. Transformer train et test avec les stats du train
X_train_clean = imputer.transform(X_train)
X_test_clean = imputer.transform(X_test)  # μ du train appliqué au test
```

---

### 🏷️ **Encodage des Variables Catégorielles**

Les algorithmes ML ne comprennent que les nombres. Pour les variables textuelles :

```python
# Exemple : type = ['PAYMENT', 'TRANSFER', 'CASH_OUT']

# Option 1 : Label Encoding (pour arbres)
le = LabelEncoder()
df['type_encoded'] = le.fit_transform(df['type'])
# Résultat : [0, 1, 2] - Ordinal implicite

# Option 2 : One-Hot Encoding (pour modèles linéaires)
df_encoded = pd.get_dummies(df, columns=['type'], drop_first=True)
# Résultat : type_TRANSFER, type_CASH_OUT (colonnes binaires)
```

**Pour Random Forest** : Label Encoding suffit (l'arbre gère naturellement les catégories).

---

## 4️⃣ ANALYSE APPROFONDIE : EXPLORATION (EDA)

### 📊 **Décrypter `.describe()`**

```python
df['amount'].describe()
```

| Statistique | Valeur | Interprétation |
|-------------|--------|----------------|
| **count** | 594,643 | Nombre de valeurs non-nulles |
| **mean** | 2,347.89 | Moyenne (centre de gravité) |
| **std** | 12,456.31 | Écart-type (dispersion) - **⚠️ ÉNORME ici** |
| **min** | 0.01 | Transaction minimale |
| **25% (Q1)** | 134.23 | 25% des transactions < 134€ |
| **50% (Médiane)** | 876.45 | Valeur centrale (robuste aux outliers) |
| **75% (Q3)** | 3,201.12 | 75% des transactions < 3,201€ |
| **max** | 10,000,000 | **🚨 OUTLIER DÉTECTÉ** |

**Analyse critique** :
- **Mean (2,347) >> Median (876)** : Distribution fortement asymétrique (skewed)
- **Std énorme** : Variance extrême causée par des transactions géantes
- **Max = 10M** : Potentiellement des fraudes ou transactions B2B exceptionnelles

**Action requise** : Transformation logarithmique pour normaliser.

```python
df['amount_log'] = np.log1p(df['amount'])  # log(1+x) pour gérer les 0
```

---

### 🔍 **La Multicollinéarité (Problème de Redondance)**

```python
# Heatmap de corrélation
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```

**Exemple typique détecté** :
- `oldbalanceOrg` ↔ `newbalanceOrig` : Corrélation = 0.98
- **Raison mathématique** : `newbalance = oldbalance - amount`

**Impact selon l'algorithme** :

| Algorithme | Impact Multicollinéarité | Action |
|------------|-------------------------|--------|
| **Random Forest** | ✅ Aucun problème | Garder toutes les variables |
| **Régression Logistique** | 🔴 Coefficients instables | Supprimer une des deux variables |
| **Réseaux de Neurones** | 🟡 Convergence plus lente | Utiliser Dropout ou PCA |

**Pour ce projet** : Random Forest étant robuste, on conserve toutes les features.

---

### 📈 **Visualisations Stratégiques**

#### **1. Distribution des montants par classe** :
```python
sns.boxplot(data=df, x='is_fraud', y='amount')
# Hypothèse : Les fraudes ont-elles des montants plus élevés ?
```

#### **2. Analyse temporelle** :
```python
df.groupby(['hour', 'is_fraud']).size().unstack().plot()
# Question : Les fraudes sont-elles plus fréquentes la nuit ?
```

#### **3. Analyse par type de transaction** :
```python
pd.crosstab(df['type'], df['is_fraud'], normalize='index')
# Question : Quel type (TRANSFER vs CASH_OUT) est le plus risqué ?
```

---

## 5️⃣ ANALYSE APPROFONDIE : MÉTHODOLOGIE (SPLIT)

### 🎲 **Le Concept : Garantie de Généralisation**

**Philosophie ML** : 
> "Le but n'est PAS de mémoriser le passé,  
> mais de PRÉDIRE sur des données JAMAIS VUES."

**Analogie** : 
- **Train Set** = Annales d'examens pour réviser
- **Test Set** = Sujet réel de l'examen (inédit)

Si on triche en révisant le sujet réel → Notes excellentes mais compétences nulles.

---

### ⚙️ **Les Paramètres sous le Capot**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # Ratio
    random_state=42,      # Reproductibilité
    stratify=y            # 🔥 CRITIQUE pour déséquilibre
)
```

#### **A. Le Ratio 80/20 (Principe de Pareto)**

| Split | % | Justification |
|-------|---|---------------|
| **Train** | 80% | Majorité pour capturer la complexité des motifs de fraude |
| **Test** | 20% | Assez grand pour être statistiquement significatif (>1000 fraudes si possible) |

**Alternative pour petits datasets** : 70/30 ou validation croisée (K-Fold).

---

#### **B. La Reproductibilité (`random_state=42`)**

```python
# Sans random_state
np.random.shuffle(data)  # Résultat différent à chaque exécution

# Avec random_state=42
np.random.seed(42)       # Graine fixe → Résultats identiques
np.random.shuffle(data)  # Toujours le même ordre
```

**Impact business** :
- ✅ Collaboration internationale : Collègue au Japon obtient mêmes résultats
- ✅ Debugging : Erreurs reproductibles
- ✅ Validation scientifique : Pairs peuvent vérifier

**Convention** : 42 est devenu le standard (référence à "Le Guide du voyageur galactique").

---

#### **C. La Stratification (`stratify=y`) - CRITIQUE POUR LA FRAUDE**

**Problème sans stratification** :

```python
# Dataset : 99% légitimes, 1% fraudes
X_train, X_test = train_test_split(X, y, test_size=0.2)

# Résultat possible (hasard malchanceux) :
# Train : 99.2% légitimes, 0.8% fraudes
# Test  : 98.5% légitimes, 1.5% fraudes

# ⚠️ Distribution différente → Modèle biaisé
```

**Solution avec stratification** :

```python
X_train, X_test = train_test_split(X, y, stratify=y)

# Résultat garanti :
# Train : 99% légitimes, 1% fraudes (exactement comme l'original)
# Test  : 99% légitimes, 1% fraudes
```

**Métaphore** : Vous voulez goûter un gâteau marbré. Sans stratification, vous risquez de tomber que sur du chocolat. Avec stratification, chaque bouchée reflète le ratio vanille/chocolat.

---

## 6️⃣ FOCUS THÉORIQUE : L'ALGORITHME RANDOM FOREST 🌲

### 🤔 **Pourquoi Random Forest pour la Fraude ?**

| Critère | Random Forest | Régression Logistique | XGBoost |
|---------|---------------|----------------------|---------|
| **Gestion non-linéarité** | ✅ Excellent | ❌ Faible | ✅ Excellent |
| **Robustesse outliers** | ✅ Très bon | ❌ Sensible | 🟡 Moyen |
| **Interprétabilité** | 🟡 Feature importance | ✅ Coefficients clairs | 🟡 Feature importance |
| **Vitesse entraînement** | 🟡 Moyenne | ✅ Rapide | ❌ Lent |
| **Gestion déséquilibre** | ✅ `class_weight='balanced'` | 🟡 Nécessite SMOTE | ✅ `scale_pos_weight` |

**Verdict** : Random Forest = "Couteau suisse" - Excellent compromis performance/simplicité.

---

### 🌳 **A. La Faiblesse de l'Individu (Arbre de Décision)**

Un arbre unique fonctionne par questions successives :

```
                    [Montant > 5000€ ?]
                     /              \
                   OUI              NON
                   /                  \
         [Type = TRANSFER ?]     [Heure > 22h ?]
           /          \            /          \
        FRAUDE    LÉGITIME    FRAUDE    LÉGITIME
```

**Problème : Haute Variance (Overfitting)**
- L'arbre mémorise le bruit : "Le client #42 avec 5,001€ à 22h01 est fraudeur"
- Sur de nouvelles données, cette règle hyper-spécifique ne marche plus
- **Performance train = 99%** / **Performance test = 75%** (overfitting)

---

### 🌲🌲🌲 **B. La Force du Groupe (Bagging)**

**Random Forest = 100 arbres diversifiés qui votent**

#### **Mécanisme 1 : Bootstrapping (Diversité des Élèves)**

```python
# Dataset original : 1000 transactions
dataset = [T1, T2, T3, ..., T1000]

# Arbre #1 s'entraîne sur un échantillon aléatoire AVEC remise
train_tree1 = random_sample(dataset, size=1000, replace=True)
# Résultat : [T42, T7, T42, T891, T7, ...]  # 42 et 7 apparaissent 2x

# Arbre #2 voit un échantillon différent
train_tree2 = random_sample(dataset, size=1000, replace=True)
# Résultat : [T3, T555, T12, T3, T910, ...]

# → Chaque arbre développe une "expertise" basée sur une expérience différente
```

---

#### **Mécanisme 2 : Feature Randomness (Diversité des Questions)**

**C'est LA magie du Random Forest.**

```python
# Dataset : 30 colonnes disponibles
# Mais à chaque nœud de l'arbre, on ne regarde que √30 ≈ 5 colonnes aléatoires

Arbre #1, Nœud racine :
  Colonnes disponibles : [amount, type, hour, oldbalance, newbalance]
  Meilleure question trouvée : "amount > 5000€ ?"

Arbre #2, Nœud racine :
  Colonnes disponibles : [merchant, day, category, balance_diff, flag]
  Meilleure question trouvée : "merchant = suspect ?"
```

**Conséquence** :
- Force les arbres à explorer des variables secondaires (texture, symétrie en médical ; merchant, timing en fraude)
- Évite que tous les arbres se focalisent sur la même variable évidente (montant)
- Réduit drastiquement la corrélation entre arbres

---

### 🗳️ **C. Le Consensus (Vote Démocratique)**

```python
# Transaction suspecte arrive
transaction_nouvelle = [amount=9500€, type=TRANSFER, hour=3h]

# Chaque arbre vote individuellement
Arbre #1  → FRAUDE
Arbre #2  → LÉGITIME
Arbre #3  → FRAUDE
...
Arbre #100 → FRAUDE

# Décompte final : 73 votes FRAUDE / 27 votes LÉGITIME
# Prédiction finale = FRAUDE (majorité)
# Probabilité = 73% de confiance
```

**Propriété mathématique magique** :
- Les erreurs individuelles (bruit) s'annulent statistiquement
- Le signal commun (vrai motif de fraude) émerge
- **Condition** : Les arbres doivent être suffisamment décorrélés (d'où le feature randomness)

---

### ⚖️ **D. Gestion du Déséquilibre : `class_weight='balanced'`**

**Problème sans ajustement** :

```python
# Dataset : 99,000 légitimes, 1,000 fraudes
model = RandomForestClassifier()
model.fit(X, y)

# Résultat : Le modèle apprend une stratégie paresseuse
# "Dire toujours LÉGITIME" → 99% d'accuracy !
# Mais 0% de fraudes détectées → Catastrophe métier
```

**Solution : Pondération des classes** :

```python
model = RandomForestClassifier(class_weight='balanced')

# Calcul automatique :
# Poids_FRAUDE = n_total / (2 * n_fraudes) = 100,000 / (2*1,000) = 50
# Poids_LÉGITIME = n_total / (2 * n_légitimes) = 100,000 / (2*99,000) ≈ 0.505

# Impact : Chaque fraude mal classée "coûte" 50x plus cher
# → Force le modèle à prioriser la détection des fraudes
```

---

## 7️⃣ ANALYSE APPROFONDIE : ÉVALUATION (L'HEURE DE VÉRITÉ)

### 📊 **A. La Matrice de Confusion (Quadrants Stratégiques)**

```
                      PRÉDICTION
                   Légitime | Fraude
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━
RÉALITÉ  Légitime |   TN    |   FP
                  |         | (Fausse Alerte)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━
          Fraude  |   FN    |   TP
                  | (Danger!)|
```

#### **Interprétation Métier** :

| Case | Nom | Signification | Impact Business | Coût |
|------|-----|---------------|-----------------|------|
| **TN** | Vrai Négatif | Légit détecté comme légit | ✅ Transaction fluide | 0€ |
| **TP** | Vrai Positif | Fraude détectée | ✅ Argent sauvé | +500€ (en moyenne) |
| **FP** | Faux Positif | Légit bloquée par erreur | 🟡 Client frustré, appel SAV | -20€ |
| **FN** | Faux Négatif | Fraude passée inaperçue | 🔴 **CATASTROPHE** | -500€ + réputation |

**Règle de décision** : 
> 1 FN coûte 25x plus cher qu'1 FP  
> → Accepter 25 FP pour éviter 1 FN

---

### 📈 **B. Les Métriques Avancées**

#### **1. Accuracy (Précision Globale) - ⚠️ PIÈGE POUR FRAUDE**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Pourquoi c'est dangereux ?**

```python
# Scénario : 1% de fraudes
# Modèle stupide qui dit toujours "LÉGITIME"

TP = 0      # Aucune fraude détectée
TN = 99,000 # Toutes les légitimes bien classées
FP = 0      # Aucune fausse alerte
FN = 1,000  # Toutes les fraudes ratées

Accuracy = (0 + 99,000) / 100,000 = 99%

# ⚠️ 99% d'accuracy mais le système est INUTILE !
```

**Verdict** : Ne JAMAIS utiliser l'accuracy seule pour des classes déséquilibrées.

---

#### **2. Precision (Qualité de l'Alarme)**

```
Precision = TP / (TP + FP)
```

**Question** : "Quand le modèle crie 'FRAUDE', a-t-il raison ?"

```python
# Exemple :
TP = 800  # 800 fraudes correctement détectées
FP = 200  # 200 fausses alertes

Precision = 800 / (800 + 200) = 0.80 = 80%

# Interprétation :
# Sur 1000 alertes générées, 800 sont de vraies fraudes
# 200 sont des fausses alarmes (clients légitimes ennuyés)
```

**Seuil acceptable** : >70% (sinon équipes anti-fraude submergées).

---

#### **3. Recall / Sensibilité (Puissance du Filet)**

```
Recall = TP / (TP + FN)
```

**Question** : "Sur toutes les fraudes réelles, combien le modèle en attrape ?"

```python
# Exemple :
TP = 800  # 800 fraudes détectées
FN = 200  # 200 fraudes ratées

Recall = 800 / (800 + 200) = 0.80 = 80%

# Interprétation :
# Sur 1000 fraudes réelles, le modèle en bloque 800
# ⚠️ 200 fraudes passent entre les mailles (coût = 200*500€ = 100,000€)
```

**Objectif métier** : >95% (laisser passer <5% de fraudes).

---

#### **4. F1-Score (Moyenne Harmonique)**

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Rôle** : Note globale qui pénalise les déséquilibres.

```python
# Cas A : Precision=0.90, Recall=0.50
F1_A = 2 * (0.90*0.50) / (0.90+0.50) = 0.64  # Médiocre

# Cas B : Precision=0.75, Recall=0.75
F1_B = 2 * (0.75*0.75) / (0.75+0.75) = 0.75  # Meilleur équilibre
```

**Usage** : Comparer deux modèles avec une seule métrique honnête.

---

#### **5. ROC AUC Score (Métrique Ultime pour Déséquilibre)**

**Concept** : Mesure la capacité du modèle à séparer les deux classes, indépendamment du seuil de décision.

```python
# Modèle parfait : AUC = 1.00 (sépare 100% des fraudes)
# Modèle aléatoire : AUC = 0.50 (pile ou face)
# Modèle acceptable : AUC > 0.90
```

**Avantage** : Robuste au déséquilibre des classes (contrairement à l'accuracy).

**Courbe ROC** :
- **Axe X** : Taux de Faux Positifs (FPR)
- **Axe Y** : Taux de Vrais Positifs (TPR = Recall)
- **Interprétation** : Plus la courbe est proche du coin supérieur gauche, meilleur est le modèle

---

### 🎯 **C. Stratégie de Seuil (Threshold Tuning)**

Par défaut, Scikit-Learn utilise `threshold=0.5` :

```python
# Proba prédite = 0.51 → FRAUDE
#

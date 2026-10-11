import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
os.makedirs("charts", exist_ok=True)
raw = pd.read_csv("titanic.csv")
df = raw.copy()
S = {}  # statistics collected for the report
 
# 1. EXPLORE
S["shape_raw"] = list(raw.shape)
S["missing_raw"] = raw.isna().sum()[lambda s: s > 0].to_dict()
S["missing_pct_raw"] = (raw.isna().mean() * 100).round(1)[lambda s: s > 0].to_dict()
S["dup_rows"] = int(raw.duplicated().sum())
S["dup_ids"] = int(raw.PassengerId.duplicated().sum())
S["name_whitespace"] = int((raw.Name != raw.Name.str.strip()).sum())
S["fare_zero"] = int((raw.Fare == 0).sum())
S["age_fractional"] = int((raw.Age.dropna() % 1 != 0).sum())
S["age_lt1"] = int((raw.Age < 1).sum())
 
def iqr_bounds(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return q1 - 1.5 * i, q3 + 1.5 * i
 
for c in ["Age", "Fare", "SibSp", "Parch"]:
    lo, hi = iqr_bounds(raw[c].dropna())
    S[f"iqr_{c}"] = {"low": round(lo, 2), "high": round(hi, 2),
                     "n_out": int(((raw[c] < lo) | (raw[c] > hi)).sum())}
S["fare_skew_raw"] = round(raw.Fare.skew(), 2)
S["fare_max"] = float(raw.Fare.max())
S["fare_gt500"] = int((raw.Fare > 500).sum())
S["embarked_missing_rows"] = raw[raw.Embarked.isna()][["PassengerId", "Pclass", "Fare", "Cabin"]].to_dict("records")
S["p1_fare_median_by_port"] = raw[raw.Pclass == 1].groupby("Embarked").Fare.median().round(2).to_dict()
S["age_mean_raw"] = round(raw.Age.mean(), 2)
S["age_std_raw"] = round(raw.Age.std(), 2)
 
# 2. CLEAN
# 2.1 text hygiene
df["Name"] = df["Name"].str.strip().str.replace(r"\s+", " ", regex=True)
 
# 2.2 Title extraction + standardisation (Mlle/Ms->Miss, Mme->Mrs, rare -> group)
df["Title"] = df.Name.str.extract(r",\s*([^.]*)\.")[0].str.strip()
S["titles_raw"] = df.Title.value_counts().to_dict()
df["Title"] = df.Title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
rare_military = ["Col", "Major", "Capt"]
rare_noble = ["Don", "Lady", "Sir", "the Countess", "Jonkheer", "Dona"]
df["Title"] = df.Title.replace(rare_military, "Officer").replace(rare_noble, "Noble")
df["Title"] = df.Title.replace({"Dr": "Professional", "Rev": "Professional"})
S["titles_clean"] = df.Title.value_counts().to_dict()
 
# 2.3 Embarked: 2 missing (same ticket, same cabin B28) -> mode imputation (Southampton, 72 %).
# Sensitivity check: 1st-class median fare by port is reported, but a single fare value is weak evidence.
S["embarked_mode"] = raw.Embarked.mode()[0]
S["embarked_mode_pct"] = round(raw.Embarked.value_counts(normalize=True)[S["embarked_mode"]] * 100, 1)
S["embarked_imputed_with"] = S["embarked_mode"]
df["Embarked"] = df["Embarked"].fillna(S["embarked_mode"])
 
# 2.4 Age: median by Title x Pclass; keep a flag so the imputation is traceable
df["Age_Imputed"] = df.Age.isna().astype(int)
grp_med = df.groupby(["Title", "Pclass"]).Age.transform("median")
title_med = df.groupby("Title").Age.transform("median")
df["Age"] = df.Age.fillna(grp_med).fillna(title_med).fillna(df.Age.median())
df["Age"] = df.Age.round(1)
S["age_mean_clean"] = round(df.Age.mean(), 2)
S["age_std_clean"] = round(df.Age.std(), 2)
S["age_median_by_title"] = df.groupby("Title").Age.median().round(1).to_dict()
 
# 2.5 Cabin: 77 % missing -> do not impute, derive Has_Cabin and Deck, drop raw column
df["Has_Cabin"] = df.Cabin.notna().astype(int)
df["Deck"] = df.Cabin.str[0].fillna("Unknown")
S["deck_counts"] = df.Deck.value_counts().to_dict()
df = df.drop(columns="Cabin")
 
# 2.6 Fare: zero fares are genuine (crew / complimentary) -> flag, do not delete
df["Fare_Zero"] = (df.Fare == 0).astype(int)
# shared tickets -> per-person fare
df["Ticket_Size"] = df.groupby("Ticket").PassengerId.transform("count")
df["Fare_Per_Person"] = (df.Fare / df.Ticket_Size).round(4)

cap = df.Fare.quantile(.99)
S["fare_cap_p99"] = round(cap, 2)
S["fare_capped_n"] = int((df.Fare > cap).sum())
df["Fare_Capped"] = df.Fare.clip(upper=cap).round(4)
df["Fare_Log"] = np.log1p(df.Fare).round(4)
S["fare_skew_capped"] = round(df.Fare_Capped.skew(), 2)
S["fare_skew_log"] = round(df.Fare_Log.skew(), 2)
 
# 2.7 Feature engineering from SibSp / Parch
df["Family_Size"] = df.SibSp + df.Parch + 1
df["Is_Alone"] = (df.Family_Size == 1).astype(int)
df["Age_Group"] = pd.cut(df.Age, [0, 12, 18, 35, 60, 120],
                         labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"], right=False)
 
# 2.8 Standardise categories / data types
df["Sex"] = df.Sex.str.strip().str.lower().map({"male": "Male", "female": "Female"})
port = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
df["Embarked"] = df.Embarked.map(port)
df["Survived"] = df.Survived.astype("int8")
df["Pclass"] = pd.Categorical(df.Pclass, categories=[1, 2, 3], ordered=True)
for c in ["Sex", "Embarked", "Title", "Deck"]:
    df[c] = df[c].astype("category")
df["Age_Group"] = df.Age_Group.astype("category")
df["Ticket"] = df.Ticket.str.strip()
 
# 3. SCALE
df["Age_MinMax"] = ((df.Age - df.Age.min()) / (df.Age.max() - df.Age.min())).round(4)
df["Fare_Log_Z"] = ((df.Fare_Log - df.Fare_Log.mean()) / df.Fare_Log.std()).round(4)
df["Family_Size_Z"] = ((df.Family_Size - df.Family_Size.mean()) / df.Family_Size.std()).round(4)
 
# 4. VALIDATE
checks = {
    "no_missing_values": bool(df.isna().sum().sum() == 0),
    "no_duplicate_rows": bool(df.duplicated().sum() == 0),
    "unique_passenger_id": bool(df.PassengerId.is_unique),
    "survived_binary": bool(df.Survived.isin([0, 1]).all()),
    "pclass_1_to_3": bool(df.Pclass.isin([1, 2, 3]).all()),
    "age_in_0_to_100": bool(df.Age.between(0.1, 100).all()),
    "fare_non_negative": bool((df.Fare >= 0).all()),
    "family_size_consistent": bool((df.Family_Size == df.SibSp + df.Parch + 1).all()),
    "age_minmax_0_1": bool(df.Age_MinMax.between(0, 1).all()),
}
S["checks"] = checks
S["shape_clean"] = list(df.shape)
S["columns_clean"] = list(df.columns)
S["survival_by_sex"] = df.groupby("Sex", observed=True).Survived.mean().round(3).to_dict()
S["survival_by_class"] = df.groupby("Pclass", observed=True).Survived.mean().round(3).to_dict()
S["survival_by_age_group"] = df.groupby("Age_Group", observed=True).Survived.mean().round(3).to_dict()
assert all(checks.values()), checks
 
df.to_csv("titanic_cleaned.csv", index=False)
 
# 5. CHARTS
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C1, C2 = "#C0504D", "#1F5C99"
 
m = raw.isna().mean().mul(100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.barh(m.index[::-1], m.values[::-1], color=[C1 if v > 0 else "#BBBBBB" for v in m.values[::-1]])
for i, v in enumerate(m.values[::-1]):
    if v > 0: ax.text(v + 1, i, f"{v:.1f}%", va="center")
ax.set_xlabel("% missing"); ax.set_title("Missing values per column (raw data)")
plt.tight_layout(); plt.savefig("charts/missing.png", dpi=170); plt.close()
 
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3), sharey=True)
ax[0].hist(raw.Age.dropna(), bins=30, color=C1); ax[0].set_title("Age - before (714 values)")
ax[1].hist(df.Age, bins=30, color=C2); ax[1].set_title("Age - after imputation (891)")
for a in ax: a.set_xlabel("Age")
plt.tight_layout(); plt.savefig("charts/age.png", dpi=170); plt.close()
 
fig, ax = plt.subplots(1, 3, figsize=(6.5, 3))
ax[0].boxplot(raw.Fare, vert=True); ax[0].set_title("Raw Fare")
ax[1].boxplot(df.Fare_Capped); ax[1].set_title("Capped (p99)")
ax[2].boxplot(df.Fare_Log); ax[2].set_title("log1p(Fare)")
for a in ax: a.set_xticks([])
plt.tight_layout(); plt.savefig("charts/fare_box.png", dpi=170); plt.close()
 
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
ax[0].hist(raw.Fare, bins=40, color=C1); ax[0].set_title(f"Fare skew = {S['fare_skew_raw']}")
ax[1].hist(df.Fare_Log, bins=40, color=C2); ax[1].set_title(f"log1p(Fare) skew = {S['fare_skew_log']}")
plt.tight_layout(); plt.savefig("charts/fare_hist.png", dpi=170); plt.close()
 
json.dump(S, open("stats.json", "w"), indent=2, default=str)
print(json.dumps(S, indent=1, default=str))

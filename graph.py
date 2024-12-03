import matplotlib.pyplot as plt
import pandas as pd

data = {
    "Language": [
        "Saint Lucian", "Seychellois", "Ndyuka", "Guadeloupean", "Haitian", "Kabuverdianu",
        "Louisiana", "Mauritian", "Papiamento", "Naija", "Tok Pisin", "Trinidadian Creole"
    ],
    "Difference": [
        2489, 1924, 843, 8012, 9701, 2433,
        2089, 2116, 6274, 1962, 1243, 4352
    ]
}

colors = [
    "#5ba8e3",
    "#1984a3", 
    "#aa79db", 
    "#b62626", 
    "#c853c6",
    "#5dbfa4", 
    "#e79742", 
    "#4f69c9", 
    "#30a319",  
    "#4fb7ac", 
    "#a5216e", 
    "#bc5dbf"
]

df = pd.DataFrame(data)

df = df.sort_values(by="Difference", ascending=False)

plt.figure(figsize=(12, 4))
plt.barh(df["Language"], df["Difference"], color=colors, height=0.95)

for index, value in enumerate(zip(df["Difference"], df["Language"])):
    plt.text(value[0] + 100, index, value[1], va='center', fontsize=10)

plt.tight_layout()
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.show()
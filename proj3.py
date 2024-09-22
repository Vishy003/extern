import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import pointbiserialr

#print("Libraries imported successfully!")

df = pd.read_excel('/Users/vishakhamishra/data/gitrepo/externship/compiled_risk_data.xlsx')
#print(df.head())

def phi_coefficient(x,y):
	contingency_table = pd.crosstab(x,y)
	chi2 = scipy.stats.chi2_contingency(contingency_table, correction=False)[0]
	n = np.sum(np.sum(contingency_table))
	phi = np.sqrt(chi2/n)
	return phi


print(df.info())
#print(df['Is_honeypot'].value_counts())

#frequencies = df['Is_honeypot'].apply(lambda x: x.value_counts()).loc[True]

# This prints out the number of True values for each risk tag.
#print(frequencies) 

risk_cols = [
	'Is_closed_source',
	'hidden_owner',
	'anti_whale_modifiable',
	'Is_anti_whale',
	'Is_honeypot',
	'buy_tax',
	'sell_tax',
	'slippage_modifiable',
	'Is_blacklisted',
	'can_take_back_ownership',
	'owner_change_balance',
	'is_airdrop_scam',
	'selfdestruct',
	'trust_list',
	'is_whitelisted',
	'is_fake_token',
	'illegal_unicode',
	'exploitation',
	'event_setter',
	'external_dependencies',
	'immutable_states',
	'reentrancy_without_eth_transfer',
	'incorrect_inheritance_order',
	'shadowing_local',
	'events_maths' ]

frequencies = df[risk_cols].apply(lambda x: x.value_counts()).loc[True]
frequencies = frequencies.fillna(0)
#print(frequencies)

sns.set_style("whitegrid")
plt.figure(figsize=(12,8))
sns.barplot(x=frequencies.index, y=frequencies.values, palette='viridis')
plt.title('Frequency of True Values for Each Risk Tag')
plt.xlabel('Risk Tags')
plt.ylabel('Frequency of True')
plt.xticks(rotation=45)
plt.show()

#correlation, p_value = pointbiserialr(df['Is_honeypot'].astype(int), df['Risk_Score'])
#print(f"Point-Biserial Correlation: {correlation}, P-value: {p_value}")

risk_df = df[risk_cols]

phi_matrix = pd.DataFrame(index=risk_df.columns, columns=risk_df.columns)

for var1 in risk_df.columns:
	for var2 in risk_df.columns:
		phi_matrix.loc[var1, var2] = phi_coefficient(risk_df[var1], risk_df[var2])
#print("Phi coefficient of all pairs of variables")
#print(phi_matrix)
#phi = phi_coefficient(df['Is_honeypot'],df['anti_whale_modifiable'])
#print(f"Phi Coefficient between 'Is_honeypot' and 'anti_whale_modifiable': {phi}")
plt.figure(figsize=(12,10))
sns.heatmap(phi_matrix.astype(float), annot=False, fmt=".2f", cmap='coolwarm', vmin=1, vmax=1)
plt.title("Heatmap of Phi Coefficient Between Risk Tags")
plt.show()

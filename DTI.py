# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Accessing data

# %% [markdown]
# ### imports

# %%
from pathlib import Path
from cloudpathlib import S3Path, S3Client
import pandas as pd



# %% [markdown]
# ### Loading preprocessed data

# %%
# Set up a local cache path
cache_path = Path('/tmp/cache')
cache_path.mkdir(exist_ok=True)

# Set up the S3 client with anonymous access
client = S3Client(local_cache_dir=cache_path, no_sign_request=True)

# Define the base S3 path
hbn_base_path_preproc = S3Path("s3://fcp-indi/", client=client)

# Define the path to the curated dataset
hbn_pod2_path = hbn_base_path_preproc / "data" / "Projects" / "HBN" / "BIDS_curated" / "derivatives"

# List the first few items in the derivatives folder
for item in hbn_pod2_path.iterdir():
    print(item)

# %% [markdown]
# ### Exploring preprocessed data

# %%
participants_table = pd.read_csv(hbn_pod2_path / "qsiprep" / "participants.tsv", sep="\t")
participants_table.head()
tract_profiles = pd.read_csv(hbn_pod2_path / "afq" / "sub-NDARAA306NT2" / "ses-HBNsiteRU" / "sub-NDARAA306NT2_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi_space-RASMM_model-CSD_desc-prob-afq_profiles.csv")

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(tract_profiles[tract_profiles["tractID"] == "CST_L"]["dki_fa"].values)
ax.set_xlabel("Node")
ax.set_ylabel("FA")

# %%
participants_table.head()

# %%
participants_table.describe()

# %%
participants_table.hist(figsize=(30,30), bins=50)

# %%
tract_profiles.head()

# %%
tract_profiles.describe()

# %% [markdown]
# ### Screen data- QC>=0.5, largest scanning site- CBIC, age>=10

# %%
# Find the most frequent scan_site_id (i.e. largest group)
largest_group = participants_table['scan_site_id'].value_counts().idxmax() #CBIC-887
filtered_participants = participants_table[(participants_table['scan_site_id'] == largest_group) & (participants_table['dl_qc_score'] >= 0.5) & (participants_table['age']>=10)]

# %%
filtered_participants.describe()

# %%
filtered_participants.head()

# %%
list(hbn_pod2_path.iterdir())
#list((hbn_pod2_path/'afq/sub-NDARZU279XR3/ses-HBNsiteCBIC').iterdir())
list((hbn_pod2_path/'afq/sub-NDARZU279XR3').iterdir())


# %%
filtered_participants['subject_id']

# %%
filtered_participants['scan_site_id'].unique()

# %% jupyter={"outputs_hidden": true}

tract_profiles_list = []

# Loop over each row in the filtered DataFrame
for _, row in filtered_participants.iterrows():
    subject_id = row['subject_id']
    scan_site = row['scan_site_id']  # Example: "CBIC", "RU", etc.
    path_ = hbn_base_path_preproc / "data" / "Projects" / "HBN" / "BIDS_curated" / "derivatives" / "afq" / f"{subject_id}" / f"ses-HBNsite{scan_site}" / f"{subject_id}_ses-HBNsite{scan_site}_acq-64dir_space-T1w_desc-preproc_dwi_space-RASMM_model-CSD_desc-prob-afq_profiles.csv"

    #df_profile=pd.read_csv(path_)
    

    
    if path_.exists():
        df_profile = pd.read_csv(path_)
        df_profile['subject_id'] = subject_id  # tag it for reference
        tract_profiles_list.append(df_profile)
        print("success")
    else:
        print(f"File not found: {path_}")
        
        


# %%

# Step 3: Concatenate into one DataFrame
#all_tract_profiles = pd.DataFrame(tract_profiles_list)
#all_tract_profiles.head()
all_tract_profiles = pd.concat(tract_profiles_list, ignore_index=True)
all_tract_profiles.describe()
#all_tract_profiles.to_csv("all_tract_profiles.csv", index=False)

# %%
'''
subject_id='NDARAA947ZG5'
scan_site='CBIC'
path_ = hbn_base_path_preproc / "data" / "Projects" / "HBN" / "BIDS_curated" / "derivatives" / "afq" / f"sub-{subject_id}" / f"ses-HBNsite{scan_site}" / f"sub-{subject_id}_ses-HBNsite{scan_site}_acq-64dir_space-T1w_desc-preproc_dwi_space-RASMM_model-CSD_desc-prob-afq_profiles.csv"


profile = pd.read_csv(path_)
'''



# %%
tract_profiles_list[1]

# %% jupyter={"outputs_hidden": true}
groups=all_tract_profiles[all_tract_profiles['subject_id']=='sub-NDARAA947ZG5'].groupby('tractID')
for tractID, df in groups:
    print (tractID, len(df))

# %%
df

# %%
profile.head()

# %%
# Save
#tract_profiles_list.to_csv("tract_profiles_list.csv", index=False)

# %% [markdown]
# # Loading filtered data from CSV

# %% jupyter={"outputs_hidden": true}
import pandas as pd
all_tract_profiles = pd.read_csv("all_tract_profiles.csv")
groups=all_tract_profiles[all_tract_profiles['subject_id']=='sub-NDARAA947ZG5'].groupby('tractID')
for tractID, df in groups:
    print (tractID, len(df))

# %% jupyter={"outputs_hidden": true}
group_by_sub=all_tract_profiles.groupby(['subject_id', 'tractID'])
for (subject_id, tractID), df in group_by_sub:
    print(subject_id, tractID, len(df))

# %%
num_tracts = all_tract_profiles['tractID'].nunique()
print(f"Number of unique tractIDs: {num_tracts}")
num_subjects=all_tract_profiles['subject_id'].nunique()
print(f"Number of filtered subjects (ages 10-25) with complete data: {num_subjects}")

# %% [markdown]
# ### Plotting 5 tracts of 5 first subjects

# %%
ordered_df = (
    all_tract_profiles
    .groupby(['subject_id', 'tractID'], sort=False)
    .apply(lambda x: x)
    .reset_index(drop=True)
)
ordered_df.head(150)

# %%
import matplotlib.pyplot as plt

# Get first 5 unique subjects
first_5_subjects = all_tract_profiles['subject_id'].unique()[:5]

# Set up plot
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 15), sharex=True, sharey=True)
fig.suptitle('First 5 Tracts of First 5 Subjects', fontsize=20)

for i, subject in enumerate(first_5_subjects):
    subject_df = all_tract_profiles[all_tract_profiles['subject_id'] == subject]
    first_5_tracts = subject_df['tractID'].unique()[:5]
    
    for j, tract in enumerate(first_5_tracts):
        tract_df = subject_df[subject_df['tractID'] == tract]
        axes[i, j].plot(tract_df['nodeID'], tract_df['dki_fa'])  # Change 'FA' to your desired metric
        axes[i, j].set_title(f'Subj: {subject}, Tract: {tract}', fontsize=10)
        axes[i, j].set_xlabel('nodeID')
        axes[i, j].set_ylabel('dki_fa')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%

# %% [markdown]
# # Building a GLM model

# %%

from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


for key, df_group in ordered_df:
    df= df_group.copy()  # copy of just this group's DataFrame
  
# 24 tract columns, 'age' column and 'sex' column

# 1. Separate features and target
x = df.drop(columns=['age'])
y = df['age']

# 2. Encode 'sex' if it's not already numeric
x['sex'] = x['sex'].map({'F': 0, 'M': 1})  # or adjust based on actual values

# 3. Split into train/test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 4. Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 5. Fit Elastic Net with CV
model = ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, .95, .99, 1], random_state=78)
model.fit(x_train_scaled, y_train)

# 6. Evaluate
print("Best alpha:", model.alpha_)
print("Best l1_ratio:", model.l1_ratio_)
print("Train R^2:", model.score(X_train_scaled, y_train))
print("Test R^2:", model.score(X_test_scaled, y_test))



# %%

# %% [markdown]
# ### Plotting acordding to age

# %%

import matplotlib.pyplot as plt

# First 5 tracts to plot
tracts_to_plot = sorted(ordered_df['tractID'].unique())[:5]

# Get all ages to loop over (sorted)
filtered_participants2 = filtered_participants.copy()
filtered_participants2['age_int'] = filtered_participants2['age'].astype(int)
ages = sorted(filtered_participants2['age_int'].unique())

for age in ages:
    # Get subjects with this age
    subjects_at_age = filtered_participants2.loc[filtered_participants2['age_int'] == age, 'subject_id']

    # Filter tract_profiles for these subjects and the chosen tracts
    subset = ordered_df[
        (ordered_df['subject_id'].isin(subjects_at_age)) &
        (ordered_df['tractID'].isin(tracts_to_plot))
    ]

    if subset.empty:
        continue  # skip if no data for this age

    # Create subplots: one row per tract
    fig, axs = plt.subplots(len(tracts_to_plot), 1, figsize=(10, 3 * len(tracts_to_plot)), sharex=True)

    for i, tract in enumerate(tracts_to_plot):
        ax = axs[i]

        tract_data = subset[subset['tractID'] == tract]

        # Plot each subject's profile as a thin line
        for subject in subjects_at_age:
            subj_data = tract_data[tract_data['subject_id'] == subject]
            if subj_data.empty:
                continue
            ax.plot(subj_data['nodeID'], subj_data['value'], alpha=0.3)

        ax.set_title(f'Tract: {tract} (Age {age})')
        ax.set_ylabel('Value')
        ax.grid(True)

    axs[-1].set_xlabel('Node along tract')

    plt.tight_layout()
    plt.show()


# %%

# Define your age bins and labels
age_bins = [10, 13, 16, 19, 23]  # 23 is exclusive upper limit
age_labels = ['10-12', '13-15', '16-18', '19-22']

# Merge age data if it's in a separate DataFrame
all_tract_profiles = all_tract_profiles.merge(filtered_participants[['subject_id', 'age']], on='subject_id')

# Add age group column
all_tract_profiles['age_group'] = pd.cut(all_tract_profiles['age'], bins=age_bins, labels=age_labels, right=False)


# %%
#tracts_to_plot = ['ATR_R', 'CST_L', 'SLF_R']  # replace with your actual choices

# %%

import matplotlib.pyplot as plt

for age_group in age_labels:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter by age group
    age_group_df = all_tract_profiles[all_tract_profiles['age_group'] == age_group]
    
    # Loop over each tract
    for tract in tracts_to_plot:
        tract_df = age_group_df[age_group_df['tractID'] == tract]
        
        # Loop over each subject in that tract
        for subject in tract_df['subject_id'].unique():
            subject_data = tract_df[tract_df['subject_id'] == subject]
            ax.plot(subject_data['nodeID'], subject_data['dki_fa'], alpha=0.3, label=f'{tract} - {subject}')
        
    ax.set_title(f'Tract profiles (dki_fa) for age group {age_group}')
    ax.set_xlabel('Node ID')
    ax.set_ylabel('DKI FA')
   # ax.legend(loc='upper right', fontsize='small', ncol=2, frameon=False)
    plt.tight_layout()
    plt.show()


# %%
import matplotlib.pyplot as plt

# Define age bins and labels
age_bins = [10, 13, 16, 19, 23]
age_labels = ['10-12', '13-15', '16-18', '19-22']

# Merge age into tract data if not already there
all_tract_profiles = all_tract_profiles.merge(
    filtered_participants[['subject_id', 'age']],
    on='subject_id'
)
all_tract_profiles['age_group'] = pd.cut(
    all_tract_profiles['age'], bins=age_bins, labels=age_labels, right=False
)

# Define tracts to plot
tracts_to_plot = ['ATR_R', 'CST_L', 'SLF_R']  # replace as needed

# Plotting
for age_group in age_labels:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for tract in tracts_to_plot:
        # Filter by age group and tract
        data = all_tract_profiles[
            (all_tract_profiles['age_group'] == age_group) &
            (all_tract_profiles['tractID'] == tract)
        ]
        
        # Pivot for mean/std by node
        mean_std = data.groupby('nodeID')['dki_fa'].agg(['mean', 'std']).reset_index()
        
        # Plot mean line
        ax.plot(
            mean_std['nodeID'],
            mean_std['mean'],
            linewidth=2.5,
            label=f'{tract} (mean)'
        )
        
        # Plot shaded std region
        ax.fill_between(
            mean_std['nodeID'],
            mean_std['mean'] - mean_std['std'],
            mean_std['mean'] + mean_std['std'],
            alpha=0.3
        )
    
    ax.set_title(f'FA Tract Profiles (Age {age_group})')
    ax.set_xlabel('Node ID')
    ax.set_ylabel('FA')
    ax.grid(True)
    # Remove legend
    ax.legend().remove()
    plt.tight_layout()
    plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# First 5 tracts to plot
tracts_to_plot = sorted(all_tract_profiles['tractID'].unique())[:5]

# Get all ages to loop over (sorted)
filtered_participants2['age_int'] = filtered_participants2['age'].astype(int)
ages = sorted(filtered_participants2['age_int'].unique())

for tract in tracts_to_plot:
    plt.figure(figsize=(10, 6))
    
    for age in ages:
        # Get subject IDs of this age
        subject_ids = filtered_participants2.loc[filtered_participants2['age_int'] == age, 'subject_id']
        
        # Get the profiles of those subjects for this tract
        subset = all_tract_profiles[(all_tract_profiles['tractID'] == tract) & 
                                    (all_tract_profiles['subject_id'].isin(subject_ids))]

        if subset.empty:
            continue
        
        # Pivot so each subject's profile is a row
        pivot = subset.pivot(index='nodeID', columns='subject_id', values='dki_fa')
        
        # Calculate mean and std across subjects
        mean_fa = pivot.mean(axis=1)
        std_fa = pivot.std(axis=1)

        # Plot with no legend
        plt.fill_between(mean_fa.index, mean_fa - std_fa, mean_fa + std_fa, alpha=0.3, linewidth=0)
        plt.plot(mean_fa.index, mean_fa, linewidth=2.5)

    plt.title(f'Tract: {tract}')
    plt.xlabel('Node')
    plt.ylabel('FA')
    plt.tight_layout()
    plt.show()

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Merge profiles and participant info
merged = all_tract_profiles.merge(
    filtered_participants[['subject_id', 'age']],
    on='subject_id',
    how='inner'
)

# Create age bins (e.g., 5-year bins)
bin_size = 5
merged['age_bin'] = pd.cut(
    merged['age'],
    bins=np.arange(merged['age'].min(), merged['age'].max() + bin_size, bin_size),
    right=False,
    include_lowest=True
)

# Select 3 tracts to plot
tracts_to_plot = sorted(merged['tractID'].unique())[:3]

# Plot
for tract in tracts_to_plot:
    plt.figure(figsize=(8, 5))
    for age_bin, group in merged[merged['tractID'] == tract].groupby('age_bin'):
        mean_profile = group.groupby('nodeID')['value'].mean()
        plt.plot(mean_profile.index, mean_profile.values, label=f'{age_bin}')
    
    plt.title(f'Tract {tract} Profile by Age Bin')
    plt.xlabel('Node ID')
    plt.ylabel('Mean Value')
    plt.legend(title='Age Bin', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Copy the participant data
filtered_participants2 = filtered_participants.copy()
filtered_participants2['age_int'] = filtered_participants2['age'].astype(int)

# Get all unique age values, sorted
ages = sorted(filtered_participants2['age_int'].unique())

# Select first 5 tract IDs for plotting
tracts_to_plot = sorted(all_tract_profiles['tractID'].unique())[:5]

# Set up the plot
fig, axs = plt.subplots(len(tracts_to_plot), 1, figsize=(12, 4 * len(tracts_to_plot)), sharex=True)

for i, tract_id in enumerate(tracts_to_plot):
    ax = axs[i]
    
    for age in ages:
        # Get subject IDs of participants with this age
        subjects = filtered_participants2.loc[filtered_participants2['age_int'] == age, 'subject_id']

        # Filter FA profiles for those subjects and the current tract
        data = all_tract_profiles[
            (all_tract_profiles['subject_id'].isin(subjects)) &
            (all_tract_profiles['tractID'] == tract_id)
        ]
        
        if data.empty:
            continue
        
        # Drop 'age_x' if it exists to avoid merge conflict
        data = data.drop(columns=[col for col in data.columns if col.startswith('age_') or col == 'age_x'], errors='ignore')
        
        # Merge with participant data to get age info
        data = data.merge(filtered_participants2[['subject_id', 'age']], on='subject_id', how='left')

        # Compute mean FA per node
        mean_fa_by_node = data.groupby('nodeID')['dki_fa'].mean()

        # Plot
        ax.plot(mean_fa_by_node.index, mean_fa_by_node.values, label=f'Age {age}')

    ax.set_title(f'Tract {tract_id}')
    ax.set_ylabel('Mean FA')
    ax.legend(title='Age')

plt.xlabel('Node ID')
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Define age bins and labels
age_bins = [10, 13, 16, 19, 23]
age_labels = ['10-12', '13-15', '16-18', '19-22']

# Remove duplicate age column if already merged before
if 'age' in all_tract_profiles.columns:
    all_tract_profiles = all_tract_profiles.drop(columns=['age'])

# Merge age info from participants
all_tract_profiles = all_tract_profiles.merge(
    filtered_participants[['subject_id', 'age']],
    on='subject_id',
    how='left'
)

# Create age group bin column
all_tract_profiles['age_group'] = pd.cut(
    all_tract_profiles['age'],
    bins=age_bins,
    labels=age_labels,
    right=False
)

# Define tracts to plot
tracts_to_plot = ['ATR_R', 'CST_L', 'SLF_R']  # You can change or expand this list

# Plotting loop
for age_group in age_labels:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for tract in tracts_to_plot:
        # Filter data by age group and tract
        data = all_tract_profiles[
            (all_tract_profiles['age_group'] == age_group) &
            (all_tract_profiles['tractID'] == tract)
        ]

        if data.empty:
            continue
        
        # Group by node and compute mean and std
        mean_std = data.groupby('nodeID')['dki_fa'].agg(['mean', 'std']).reset_index()

        # Plot mean line
        ax.plot(
            mean_std['nodeID'],
            mean_std['mean'],
            linewidth=2.5,
            label=f'{tract}'
        )

        # Plot std band (shaded)
        ax.fill_between(
            mean_std['nodeID'],
            mean_std['mean'] - mean_std['std'],
            mean_std['mean'] + mean_std['std'],
            alpha=0.4
        )
    
    # Title and axis settings
    ax.set_title(f'dki_fa Tract Profiles (Age {age_group})')
    ax.set_xlabel('Node ID')
    ax.set_ylabel('dki_fa')
    ax.grid(True)

    # Remove legend (optional: keep if needed)
    ax.legend().remove()

    plt.tight_layout()
    plt.show()

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Define age bins and labels
age_bins = [10, 13, 16, 19, 23]
age_labels = ['10-12', '13-15', '16-18', '19-22']

# Remove duplicate age column if already merged before
if 'age' in all_tract_profiles.columns:
    all_tract_profiles = all_tract_profiles.drop(columns=['age'])

# Merge age info from participants
all_tract_profiles = all_tract_profiles.merge(
    filtered_participants[['subject_id', 'age']],
    on='subject_id',
    how='left'
)

# Create age group bin column
all_tract_profiles['age_group'] = pd.cut(
    all_tract_profiles['age'],
    bins=age_bins,
    labels=age_labels,
    right=False
)

# Define tracts to plot
tracts_to_plot = ['ATR_R', 'CST_L', 'SLF_R']  # You can change or expand this list

# Plotting loop
for age_group in age_labels:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for tract in tracts_to_plot:
        # Filter data by age group and tract
        data = all_tract_profiles[
            (all_tract_profiles['age_group'] == age_group) &
            (all_tract_profiles['tractID'] == tract)
        ]

        if data.empty:
            continue
        
        # Group by node and compute mean and std
        mean_std = data.groupby('nodeID')['dki_fa'].agg(['mean', 'std']).reset_index()

        # Plot mean line
        ax.plot(
            mean_std['nodeID'],
            mean_std['mean'],
            linewidth=2.5,
            label=f'{tract}'
        )

        # Plot std band (shaded)
        ax.fill_between(
            mean_std['nodeID'],
            mean_std['mean'] - mean_std['std'],
            mean_std['mean'] + mean_std['std'],
            alpha=0.4
        )
    
    # Title and axis settings
    ax.set_title(f'FA Tract Profiles (Age {age_group})')
    ax.set_xlabel('Node ID')
    ax.set_ylabel('FA')
    ax.grid(True)
    ax.legend(title='Tract')
    ax.set_ylim(0.2, 0.6)
    # Remove legend (optional: keep if needed)
    #ax.legend().remove()
    
    plt.tight_layout()
    plt.show()

# %%

# %%

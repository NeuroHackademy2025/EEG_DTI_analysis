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
filtered_participants.head(5000)

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

# %%
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
ordered_df.head(5000)

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

# %% [markdown]
# ### Plotting acordding to age

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
ordered_df2=ordered_df.copy()
ordered_df_age = ordered_df2.merge(
    filtered_participants[['subject_id', 'age']],
    on='subject_id',
    how='left'
)
ordered_df_age.head(2000)

# %%
ordered_df2 = ordered_df.drop(columns=['age'], errors='ignore')  # remove 'age' if it exists
ordered_df_age = ordered_df2.merge(
    filtered_participants[['subject_id', 'age','sex']],
    on='subject_id',
    how='left'
)
ordered_df_age.head(5000)

# %%
'''
# 1. Drop unwanted columns
df=ordered_df_age.copy()

# 2. Average dki_fa per subject and tract (mean across nodes)
fa_avg = df.groupby(['subject_id', 'tractID'])['dki_fa'].mean().reset_index()

# 3. Pivot so each tract becomes a column, rows are subjects
fa_wide = fa_avg.pivot(index='subject_id', columns='tractID', values='dki_fa')

# Extract sex per subject (unique value)
sex_df = df[['subject_id', 'sex','age']].drop_duplicates().set_index('subject_id')

# Join sex to the fa_wide dataframe
fa_wide = fa_wide.join(sex_df)

# Now fa_wide has sex as an additional column
fa_wide.head()
'''

# %%
# 1. Drop unwanted columns
df = ordered_df_age.copy()

# 2. Average dki_fa per subject and tract (mean across nodes)
fa_avg = df.groupby(['subject_id', 'tractID'])['dki_fa'].mean().reset_index()

# 3. Pivot so each tract becomes a column, rows are subjects
fa_wide = fa_avg.pivot(index='subject_id', columns='tractID', values='dki_fa')

# Extract sex and age per subject (unique value)
sex_df = df[['subject_id', 'sex', 'age']].drop_duplicates().set_index('subject_id')

# Convert age to int before joining
sex_df['age'] = sex_df['age'].astype(int)

# Join sex and age to the fa_wide dataframe
fa_wide = fa_wide.join(sex_df)

# Now fa_wide has sex and age as additional columns
fa_wide.head()

# %%
print(fa_wide.columns)

# %% [markdown]
# # Building a GLM model

# %%
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Prepare features X and target y
fa_wide=fa_wide.dropna()
X = fa_wide.drop(columns=['age'])
y = fa_wide['age'].astype(int)

categorical_cols = ['sex']

# One-hot encode categorical column 'sex'
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='if_binary'), categorical_cols)
    ],
    remainder='passthrough'  # keep numeric columns as-is
)


# Create pipeline with preprocessing and LinearRegression
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Evaluate on test set
r2 = model.score(X_test, y_test)
print(f'Test R^2 score: {r2:.3f}')


# %%

# %%

# Predict ages for the test set
y_pred = model.predict(X_test)

# Scatter plot: actual vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Actual vs. Predicted Age')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Adding regularization

# %%
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Prepare data
fa_wide = fa_wide.dropna()
X = fa_wide.drop(columns=['age'])
y = fa_wide['age'].astype(int)

categorical_cols = ['sex']

# Preprocessing: One-hot encode 'sex'
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='if_binary'), categorical_cols)
    ],
    remainder='passthrough'
)

# Define models to test
models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate each model
for name, regressor in models.items():
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f'{name} R² score: {r2:.3f}')

# %%

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
'''
subject_id='sub-NDARAC331VEH'
scan_site='CBIC'
path_ = hbn_base_path_preproc / "data" / "Projects" / "HBN" / "BIDS_curated" / "derivatives" / "afq" / f"{subject_id}" / f"ses-HBNsite{scan_site}" / f"{subject_id}_ses-HBNsite{scan_site}_acq-64dir_space-T1w_desc-preproc_dwi_space-RASMM_model-CSD_desc-prob-afq_profiles.csv"


profile = pd.read_csv(path_)

profile.head()

'''

# %%
filtered_participants['scan_site_id'].unique()

# %%

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
'''
for _, row in filtered_participants.iterrows():
    subject_id = row['subject_id']  # or the correct column name
    scan_site = row['scan_site_id']

    
    # Build path to tract_profiles file
    file_path = (
        hbn_pod2_path / "afq" / f"sub-{subject_id}" / f"ses-HBNsite{scan_site}" /
        f"sub-{subject_id}_ses-HBNsite{scan_site}_acq-64dir_space-T1w_desc-preproc_dwi_space-RASMM_model-CSD_desc-prob-afq_profiles.csv"
    )

    if file_path.exists():
        df_profile = pd.read_csv(file_path)
        df_profile['subject_id'] = subject_id  # tag it for reference
        tract_profiles_list.append(df_profile)
    else:
        print(f"File not found: {file_path}")

'''



# %%

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

# %%
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

# %%

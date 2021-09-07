SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly',
          'https://www.googleapis.com/auth/drive.readonly',
          'https://www.googleapis.com/auth/drive.metadata.readonly']

spreadsheets = {'session_code_map': {'spreadsheet': 'SubjectSessionMapping', 'sheet': 'SubjectSessionMapping'},
                'participants': {'spreadsheet': 'ParticipantExport', 'sheet': 'ParticipantExport'},
                'sessions': {'spreadsheet': 'SessionExport', 'sheet': 'SessionExport'},
                'mds_updrs': {'spreadsheet': 'NEW PD Progression clinical data ', 'sheet': 'MDS_UPDRS'},
                'old_updrs': {'spreadsheet': 'NEW PD Progression clinical data ', 'sheet': 'Old_UPDRS'},
                'hads': {'spreadsheet': 'NEW PD Progression clinical data ', 'sheet': 'HADS'},
                'meds': {'spreadsheet': 'NEW PD Progression clinical data ', 'sheet': 'Medication'},
                'neuropsyc': {'spreadsheet': 'RedcapExport', 'sheet': 'RedcapExport'},
                'mri': {'spreadsheet': 'PD Scan numbers', 'sheet': 'RedcapExport'},}

# Attempt re-loading data if the spreadsheet was modified within the last XX seconds
time_after_update_wait = 30

# Seconds to wait before attempting to reload data.
wait_for_update = 10

# How many attempts to make before throwing an error.
retry_attempts = 10

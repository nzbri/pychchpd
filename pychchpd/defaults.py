SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly',
          'https://www.googleapis.com/auth/drive.readonly',
          'https://www.googleapis.com/auth/drive.metadata.readonly']

spreadsheets = {'session_code_map': {'spreadsheet': 'SubjectSessionMapping', 'sheet': 'SubjectSessionMapping', 'key': '1JAmxvXPU0cQlQAlYiZUwCxk1oDTu_84w4ZZUYZo2VP4'},
                'participants': {'spreadsheet': 'ParticipantExport', 'sheet': 'ParticipantExport', 'key': '1WmeDr5WbUzl2uA1wMlZTJdcn_F-q-nWMEmlCKjSFInk'},
                'sessions': {'spreadsheet': 'SessionExport', 'sheet': 'SessionExport', 'key': '1HS_wlXRbWmGV3Db8ELuB53N6O1xVSmWJqlJ0AhyaMdY'},
                'mds_updrs': {'spreadsheet': 'NEW PD Progression clinical data ', 'sheet': 'MDS_UPDRS', 'key': '14Jb3qC1Ioazmpacpwtlqw-myFtIjBGR9D2y6znLVIbs'},
                'old_updrs': {'spreadsheet': 'NEW PD Progression clinical data ', 'sheet': 'Old_UPDRS', 'key': '14Jb3qC1Ioazmpacpwtlqw-myFtIjBGR9D2y6znLVIbs'},
                'hads': {'spreadsheet': 'NEW PD Progression clinical data ', 'sheet': 'HADS', 'key': '14Jb3qC1Ioazmpacpwtlqw-myFtIjBGR9D2y6znLVIbs'},
                'meds': {'spreadsheet': 'NEW PD Progression clinical data ', 'sheet': 'Medication', 'key': '14Jb3qC1Ioazmpacpwtlqw-myFtIjBGR9D2y6znLVIbs'},
                'neuropsyc': {'spreadsheet': 'RedcapExport', 'sheet': 'RedcapExport', 'key': '1liI06efJe1mRI3Iz2lWwq_PeZDpZLnIhhjqyN1SQlX0'},
                'mri': {'spreadsheet': 'PD Scan numbers', 'sheet': 'RedcapExport', 'key': '1NVlN6GrzXuyP4u37iuO2NAfORJxC8bmaBWIZpPoyrR4'},
                'assessments': {'spreadsheet': 'AssessmentExport', 'sheet': 'AssessmentExport', 'key': '1FCsQWiMJFFXdJ4qcnzyGhWoplIpZ4N4QD7ijKdmNgMo'},}

# Attempt re-loading data if the spreadsheet was modified within the last XX seconds
time_after_update_wait = 30

# Seconds to wait before attempting to reload data.
wait_for_update = 10

# How many attempts to make before throwing an error.
retry_attempts = 10

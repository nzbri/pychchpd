# pychchpd

Work in progress, might be buggy at times.

Install via pip:

```
pip install git+https://github.com/rezashr/pychchpd
```

Here's an example:

```python
# Import package.
import pychchpd

# Initiate an instance of chchpd. This will perform google authentication.
chchpd = pychchpd.chchpd(use_server=False)

# Set whether to use anon_id or subject_id
chchpd.anonymize(True) # Use anon_id.

# Set verbose
chchpd.set_verbose(verbose=True)

# Choose whether to include hyphen and underscore in session ID. 
# This is particularly relevant to BIDS structure, where these characters are not allowed. 
chchpd.session_id(allow_hyphen=True, allow_underscore=True)

# Perform data import, similar to that of CHCHPD.
sessions = chchpd.import_sessions()
participants = chchpd.import_participants()

# Converting between anon_id and subject_id
subject_id = chchpd.get_subject_id_from_anon_id(anon_id=anon_id)
anon_id = chchpd.get_anon_id_from_subject_id(subject_id=subject_id)
```
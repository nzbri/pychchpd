import warnings
import gspread
import pandas as pd
import numpy as np
from . import defaults
from .utils import return_copy
from pathlib import Path
from functools import lru_cache
from dateutil.parser import parse
from datetime import datetime
from gspread_dataframe import get_as_dataframe
import janitor
from tabulate import tabulate
import time

cache = lru_cache(maxsize=None)


def move_columns(df, cols_to_move: list, ref_col: str, place='After'):
    """
    A helper function to move columns in pandas DataFrames.
    From:
      https://towardsdatascience.com/reordering-pandas-dataframe-columns-thumbs-down-on-standard-solutions-1ff0bc2941d5

    Parameters:
        df: pandas DataFrame.

        cols_to_move: A list of column-names to move.

        ref_col: Reference column.

        place: Either 'after' or 'before' (case insensitive).

    Returns:
        A new DataFrame with columns moved.
    """
    cols = df.columns.tolist()
    place = place.lower()

    if isinstance(ref_col, list):
        ref_col = ref_col[0]

    if place == 'after':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    elif place == 'before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    else:
        raise ValueError(f'I cannot understand place={place}.')

    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]

    return df[seg1 + seg2 + seg3]


def _fix_nas(data, na=None):
    """
    This function removes `na` strings in a case-insensitive manner.

    Parameters:
        data: A pandas data-frame.

        na: List of values to be replaced with NA.

    Returns:
        A data-frame with NA value for elements of `na`.
    """
    if not na:
        return data.fillna(np.nan)

    na = ['(?i)^' + x.lower() + '$' if isinstance(x, str) else x for x in na]
    data = data.replace(na, pd.NA, regex=True)
    return data.fillna(np.nan)


def _fix_column_names(data):
    data = data.copy()
    data.columns = data.columns.str.replace('part_iii', 'Part_III', regex=True)
    data.columns = data.columns.str.replace('part_ii', 'Part_II', regex=True)
    data.columns = data.columns.str.replace('part_i', 'Part_I', regex=True)
    data.columns = data.columns.str.replace('h_y', 'H_Y', regex=True)
    data.columns = data.columns.str.replace('^q(\d.*)', r'Q\1', regex=True)
    data.columns = data.columns.str.replace('updrs', 'UPDRS', regex=True)
    data.columns = data.columns.str.replace('mds', 'MDS', regex=True)
    data.columns = data.columns.str.replace('hads', 'HADS', regex=True)
    return data.fillna(np.nan)


def _google_authenticate(use_server=True, scopes=None):
    """
    Perform google authenticate.

    Parameters:
        use_server: If True, use the copy/paste authentication process. Otherwise open a browser to perform
        authentication.

        scopes: The scope(s) to access google drive. By default, `google_authenticate` uses read-only scopes for
        spreadsheet, drive, and metadata APIs.

    Returns:
        An instance of gspread to find and open spreadsheets.
    """
    oauth_file = str(Path(__file__).parent.joinpath('oauth.json'))
    token_dir = Path.home().joinpath('.pychchpd')
    token_dir.mkdir(exist_ok=True, parents=True)
    token_file = token_dir.joinpath('pychchpd.json').__str__()

    if use_server:
        flow = gspread.auth.console_flow
    else:
        flow = gspread.auth.local_server_flow

    if scopes is None:
        scopes = defaults.SCOPES

    return gspread.oauth(
        flow=flow,
        scopes=scopes,
        credentials_filename=oauth_file,
        authorized_user_filename=token_file)


# Here is the CHCHPD class.
class chchpd:
    def __init__(self, use_server=True, scopes=None):
        self._use_server = use_server
        if scopes is None:
            self.scopes = defaults.SCOPES
        else:
            self.scopes = scopes

        self._anonymize_id = True
        self._session_id_allow_hyphen = True
        self._session_id_allow_underscore = True

        self.__gc = _google_authenticate(use_server=self._use_server,
                                         scopes=self.scopes)

        self._cached_data = {}
        self._verbose = True

    def __repr__(self):
        return f'CHCHPD Package - Anonymize={self._anonymize_id}, Verbose={self._verbose}.'

    def anonymize(self, use_anon_id):
        self._anonymize_id = bool(use_anon_id)

    def set_verbose(self, verbose):
        self._verbose = bool(verbose)

    @property
    def verbose(self):
        return self._verbose

    def session_id(self, allow_hyphen=True, allow_underscore=True):
        self._session_id_allow_hyphen = bool(allow_hyphen)
        self._session_id_allow_underscore = bool(allow_underscore)

    def _get_subject_id_anon_id_conversion_dict(self):
        participant_conversion_table = (
            self._load_spreadsheet('participants', na=['NA', '', 'None', None], fix_anon=False).
            filter(items=['subject_id', 'anon_id']))

        participant_conversion_dict = dict(zip(participant_conversion_table.subject_id,
                                               participant_conversion_table.anon_id))
        return participant_conversion_dict

    def get_anon_id_from_subject_id(self, subject_id):
        participant_conversion_dict = self._get_subject_id_anon_id_conversion_dict()
        return participant_conversion_dict.get(subject_id)

    def get_subject_id_from_anon_id(self, anon_id):
        participant_conversion_dict = self._get_subject_id_anon_id_conversion_dict()
        anon_conversion_dict = {anon: subid for subid, anon in participant_conversion_dict.items()}
        return anon_conversion_dict.get(anon_id)

    def _fix_subject_id(self, dataset):
        if not self._anonymize_id:
            return dataset.fillna(np.nan)

        def _identify_col(orig_col):
            cols_of_interest = ['_id', '_label']
            for col in cols_of_interest:
                if col in orig_col and orig_col != 'survey_id':
                    return True
            return False

        cols = [col for col in dataset.columns if _identify_col(col)]
        participant_conversion_dict = self._get_subject_id_anon_id_conversion_dict()

        for col in cols:
            # These columns should be string. !!!
            dataset[col] = dataset[col].astype(str)

            # Split based on '_'. This assumes that subject IDs don't have underscore.
            tmp_col = dataset[col].str.split('_', expand=True)

            # Replace subject_id with anon_id.
            tmp_col[0] = tmp_col.apply(lambda x: participant_conversion_dict.get(x[0], x[0]), 1)

            dataset[col] = tmp_col.apply(lambda x: '_'.join(list(filter(lambda y: y is not None, x))), 1)
        return dataset.fillna(np.nan)

    def _map_to_universal_session_id(self, dataset, make_session_label=True, remove_double_measures=True,
                                     drop_original_columns=True):

        dataset = dataset.copy()  # Ensure that pandas does not change the original data in anyway.
        session_code_map = self._load_spreadsheet('session_code_map')
        session_code_map = session_code_map.rename(columns={'session_id': 'session_suffix'})
        session_code_map['input_id'] = session_code_map.apply(lambda x: f'{x.subject_id}_{x.session_suffix}', 1)
        session_code_map = session_code_map.drop(columns=['subject_id', 'session_suffix'])

        def check_date(x):
            try:
                _ = pd.to_datetime(x)
                return False
            except:
                return True

        session_code_map['incorrect_rows'] = session_code_map.apply(lambda x: check_date(x.standardised_session_id), 1)

        # Fixme: Lets remove the incorrect session dates. Hopefully this gets fixed in the next release of redcap
        #  export.
        session_code_map['session_id'] = session_code_map.apply(
            lambda x: f'{x.standardised_subject_id}_{x.standardised_session_id}' if not x.incorrect_rows else '', 1)

        if not self._session_id_allow_hyphen:
            session_code_map['session_id'] = session_code_map['session_id'].replace('-', '', regex=True)

        if not self._session_id_allow_underscore:
            session_code_map['session_id'] = session_code_map['session_id'].replace('_', '', regex=True)

        session_code_map = session_code_map.drop(
            columns=['standardised_session_id', 'standardised_subject_id', 'incorrect_rows'])

        if make_session_label:
            dataset['session_label'] = dataset.apply(
                lambda x: f'{x.subject_id}_{x.session_suffix}', 1)

            if drop_original_columns:
                dataset = dataset.drop(columns=['subject_id', 'session_suffix'])

        unmatched = (pd.merge(left=dataset, right=session_code_map,
                              left_on='session_label', right_on='input_id', how='left', indicator=True).
                     query('_merge=="left_only"').filter(items=dataset.columns))

        dataset = pd.merge(left=dataset, right=session_code_map, left_on='session_label',
                           right_on='input_id', how='left')

        dataset = move_columns(df=dataset, cols_to_move=['session_id'],
                               ref_col=dataset.columns[0], place='Before')

        if len(unmatched) > 0 and self.verbose:
            print('Records failed to match to a universal session ID:')
            print(tabulate(unmatched, headers='keys', tablefmt='fancy_grid'))

        if remove_double_measures:
            date_cols = dataset.columns[dataset.columns.str.endswith('date')].to_list()
            dataset = (dataset.groupby(['session_id'], sort=False, group_keys=False).
                       apply(lambda x: x.sort_values(date_cols, ascending=False)))

            dataset = dataset.groupby('session_id', sort=False).head(1).reset_index(drop=True)

        if make_session_label and drop_original_columns:
            dataset = dataset.drop(columns='session_label')
        return dataset.fillna(np.nan)

    @return_copy
    @cache
    def _get_spreadsheet_info(self, title=None, key=None):
        if key:
            tmpinfo = self.__gc.open_by_key(key)
            info = dict(
                id=tmpinfo.id,
                name=tmpinfo.title,
                createdTime=tmpinfo.creationTime,
                modifiedTime=tmpinfo.lastUpdateTime
            )
            return info

        info = self.__gc.list_spreadsheet_files(title)

        if len(info) == 0:
            raise ValueError(f'{title} spreadsheet does not exist or you do not have permission to access it.')
        elif len(info) > 1:
            warnings.warn(
                f'There are more than one {title} spreadsheet. I will load the first one for now, but this may lead '
                f'to inaccurate data.')

        return info[0]

    def _load_data_fresh(self, spreadsheet_id, sheet, na=None):
        sh = self.__gc.open_by_key(spreadsheet_id).worksheet(sheet)
        data = get_as_dataframe(sh, evaluate_formulas=True, skipinitialspace=True, na_values=na,
                                parse_dates=True, keep_date_col=True, infer_datetime_format=False)

        data = data.clean_names(case_type='snake', remove_special=True)

        # Remove empty columns and rows.
        data = data.dropna(how='all')
        # data = data.dropna(how='all', axis='columns')
        data = data.drop_duplicates()
        data = data.infer_objects()

        # Check if there are date columns, and convert them to proper date.
        date_columns = data.columns.str.contains('_date')
        if date_columns.any():
            data.loc[:, date_columns] = data.loc[:, date_columns].astype('datetime64')

        return data.fillna(np.nan)

    def _load_data(self, modality, info=None, na=None):
        title = defaults.spreadsheets[modality]['spreadsheet']
        sheet = defaults.spreadsheets[modality]['sheet']
        key = defaults.spreadsheets[modality]['key']

        if info is None:
            info = self._get_spreadsheet_info(title=title, key=key)

        info = info.copy()
        spreadsheet_id = info['id']
        load_required = True

        cached_info = self._cached_data.get(modality)
        # Check whether this spreadsheet has been cached already.
        if cached_info and cached_info['modifiedTime']:
            if cached_info['modifiedTime'] == info['modifiedTime']:
                load_required = False

        if load_required:
            data = self._load_data_fresh(spreadsheet_id, sheet, na=na)
            data = _fix_nas(data, na)
            info.update({'cached_data': data})
            self._cached_data.update({modality: info})
        else:
            data = self._cached_data[modality]['cached_data']

        return data.fillna(np.nan).copy()

    def _load_spreadsheet(self, modality, na=None, fix_anon=True):
        title = defaults.spreadsheets[modality]['spreadsheet']
        key = defaults.spreadsheets[modality]['key']
        accept_spreadsheet = False
        info = ()
        for attempt in range(max(0, defaults.retry_attempts)):
            info = self._get_spreadsheet_info(title=title, key=key)
            modified_time = parse(info.get('modifiedTime'))
            elapsed_time_since_update = (datetime.now(tz=modified_time.tzinfo) - modified_time).total_seconds()
            if (elapsed_time_since_update < defaults.time_after_update_wait):
                print(f'Attempt #{attempt}/{defaults.retry_attempts}: {modality} spreadsheet is updating.')
                time.sleep(defaults.wait_for_update)
            else:
                accept_spreadsheet = True

        if not accept_spreadsheet:
            raise ValueError(f'Could not load {modality} spreadsheet, please check.')

        data = self._load_data(modality=modality, info=info, na=na)
        if fix_anon:
            return self._fix_subject_id(data)
        else:
            return data.fillna(np.nan)

    def import_participants(self, identifiers=False):
        data = self._load_spreadsheet(modality='participants',
                                      na=['NA', '', 'None', None])

        # Fix date columns. This converts columns to numpy.datetime64[ns]
        data = data.astype({'birth_date': 'datetime64', 'date_of_death': 'datetime64', 'survey_id': 'Int64'})
        data = data.astype({'survey_id': str})

        data = data.rename(columns=dict(status='participant_status',
                                        excluded='excluded_from_followup',
                                        disease_group='participant_group',
                                        side_affected='side_of_onset'))

        data['side_of_onset'] = data.apply(
            lambda x: pd.NA if isinstance(x.side_of_onset, str) and x.side_of_onset == 'Unknown' else x.side_of_onset,
            1)
        data['dead'] = data.apply(lambda x: not pd.isna(x.date_of_death), 1)
        data = (data.assign(handedness=pd.Categorical(data['handedness']),
                            side_of_onset=pd.Categorical(data['side_of_onset']),
                            sex=pd.Categorical(data['sex'], categories=['Male', 'Female']),
                            participant_status=pd.Categorical(data['participant_status'])))

        time_unit = np.timedelta64(1, 'D')

        def calc_age_today(x):
            if pd.isna(x.birth_date):
                return pd.NaT
            return np.round(
                (datetime.today() - x.birth_date).to_timedelta64().astype(time_unit).astype(float) / 365.25, 2)

        data['age_today'] = data.apply(lambda x: calc_age_today(x), 1)

        def get_age_at_death(x):
            if pd.isna(x.date_of_death):
                return np.nan
            age = (x.date_of_death - x.birth_date).to_timedelta64().astype(time_unit).astype(float) / 365.25
            return age

        data['age_at_death'] = data.apply(lambda x: np.round(get_age_at_death(x), 2), 1)

        if identifiers:
            data = data.filter(['subject_id', 'survey_id', 'first_name', 'last_name', 'date_of_death',
                                'dead', 'participant_status', 'birth_date', 'age_today',
                                'excluded_from_followup', 'participant_group', 'sex',
                                'side_of_onset', 'handedness', 'symptom_onset_age', 'diagnosis_age',
                                'education', 'ethnicity'])

        else:
            data = data.filter(['subject_id', 'survey_id', 'dead', 'participant_status',
                                'excluded_from_followup',
                                'participant_group', 'sex', 'side_of_onset', 'handedness',
                                'symptom_onset_age', 'diagnosis_age', 'age_at_death', 'age_today',
                                'education', 'ethnicity'])

        # tabulate_duplicates(participants, 'subject_id')

        return data.fillna(np.nan)

    def import_sessions(self, from_study=None, exclude=True):
        data = self._load_spreadsheet(modality='sessions', na=('', 'None', None))

        data = data.astype({'date': 'datetime64', 'mri_scan_no': 'UInt64'}).astype({'mri_scan_no': str})
        data['mri_scan_no'] = data.apply(lambda x: '' if x.mri_scan_no == str(pd.NA) else x.mri_scan_no, 1)

        data['session_suffix'] = data.session_id.str.split('_', expand=True)[1]
        data = (data.filter(['session_id', 'subject_id', 'session_suffix', 'study', 'date',
                             'study_group', 'study_excluded', 'mri_scan_no']).
                rename(columns={'session_id': 'session_label',
                                'date'      : 'session_date'}).
                assign(study=pd.Categorical(data['study'])))

        data = self._map_to_universal_session_id(data, make_session_label=False, remove_double_measures=False)

        if exclude:
            data['selection_idx'] = data.apply(
                lambda x: (pd.isna(x.study_excluded) == False) & bool(x.study_excluded == True), 1)

            excluded_session = data.query('selection_idx==True').reset_index(drop=True).drop(
                columns=['selection_idx'])

            if self.verbose and len(excluded_session) > 0:
                print('Automatically excluded sessions:')
                exclusion_summary = (excluded_session.groupby('study')['study'].count().to_frame().
                                     query('study != 0').rename(columns={'study': 'Excluded'}))
                print(tabulate(exclusion_summary, headers='keys', tablefmt='fancy_grid'))

            data['selection_idx'] = data.apply(lambda x: pd.isna(x.study_excluded) | bool(x.study_excluded != True), 1)
            data = data.query('selection_idx').reset_index(drop=True).drop(columns=['selection_idx'])

        if from_study is not None:
            if isinstance(from_study, str) or not hasattr(from_study, '__iter__'):
                from_study = [from_study]

            data = data.query(f'study in {from_study}')

        birthdate = self.import_participants(identifiers=True).filter(items=['subject_id', 'birth_date'])

        data = (pd.merge(left=data, right=birthdate, on='subject_id', how='left').
                assign(age=lambda x: np.round((x.session_date - x.birth_date).to_numpy().
                                              astype('timedelta64[D]').astype('float') / 365.25, 2)))

        data = data.loc[:, 'session_id':'session_date'].join(data.loc[:, 'age']).join(
            data.loc[:, 'study_group':'mri_scan_no'])

        data = data.groupby('subject_id', sort=False, group_keys=False).apply(
            lambda x: x.sort_values(['subject_id', 'session_date'], ascending=True)).reset_index(drop=True)
        return data.fillna(np.nan)

    def import_mds_updrs(self, concise=True):
        data = self._load_spreadsheet(modality='mds_updrs',
                                      na=['na', 'NA', 'NA1', 'NA2', 'NA3', 'NA4', 'NA5', 'NA6', 'UR', '', 'None', None])

        data = self._map_to_universal_session_id(data)

        for col in data.columns[data.columns.str.match(r'(^q\d.*)')]:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        data = data.assign(h_y_missing=(data['h_y_missing'] == 'Y'),
                           part_iii_na=(data['part_iii_missing'].isna()),
                           part_i=data.loc[:, 'q1_1':'q1_13'].sum(axis=1, skipna=True),
                           part_i_missing=data.loc[:, 'q3_1':'q3_18'].isna().mean(axis=1).round(),
                           part_ii=data.loc[:, 'q2_1':'q2_13'].sum(axis=1, skipna=True),
                           part_ii_missing=data.loc[:, 'q2_1':'q2_13'].isna().mean(axis=1).round(),
                           part_iii=data.loc[:, 'q3_1':'q3_18'].sum(axis=1, skipna=True),
                           part_iii_missing=data.loc[:, 'q3_1':'q3_18'].isna().mean(axis=1).round()
                           )

        data['part_i'] = data.apply(lambda x: pd.NA if x.part_i_missing == 1 else x.part_i, 1)
        data['part_ii'] = data.apply(lambda x: pd.NA if x.part_ii_missing == 1 else x.part_ii, 1)
        data['part_iii'] = data.apply(lambda x: pd.NA if x.part_iii_missing == 1 else x.part_iii, 1)

        # Todo: Check me please...
        data = data.query('not(h_y.isna() and part_iii_na)').drop(columns=['h_y_missing', 'part_iii_na'])

        if concise:
            cols_to_keep = ['session_id', 'h_y'] + data.columns[
                data.columns.slice_indexer('part_i', 'part_iii')].to_list()
            data = data.filter(items=cols_to_keep)

        return _fix_column_names(data)

    def import_old_updrs(self):
        data = self._load_spreadsheet('old_updrs',
                                      na=['na', 'NA', 'NA1', 'NA2', 'NA3', 'NA4', 'NA5', 'NA6', 'UR', '', None])

        # remove 2nd sessions from Charlotte Graham's amantadine follow-up study,
        # where there wasn't a full NP session done, as the session was only a few
        # months after the baseline session:
        data = data.query('full_neuropsych=="y"')
        data = self._map_to_universal_session_id(data)

        # calculate part scores for the UPDRS
        # (non-motor, ADL, motor).
        # Here, we make references using column names rather than numbered ranges
        # e.g. Q1_1 to Q1_13 instead of [6:18] for Part I, to make things less
        # brittle if columns are added to the source spreadsheet, but this requires
        # some jiggery pokery to get the indices to the numbered columns

        data = data.assign(part_i_raw=data.loc[:, 'q1':'q4'].sum(axis=1, skipna=True),
                           part_ii_raw=data.loc[:, 'q5':'q17'].sum(axis=1, skipna=True),
                           part_iii_raw=data.loc[:, 'q18':'q31'].sum(axis=1, skipna=True))

        def comp_mds_part_ii(h_y, part_ii_raw):
            if pd.isna(h_y):
                return pd.NA
            elif h_y < 3:
                return part_ii_raw * 1.1 + 0.2
            elif h_y < 4:
                return part_ii_raw * 1.0 + 1.5
            elif h_y <= 5:
                return part_ii_raw * 1.0 + 4.7
            else:
                return pd.NA

        def comp_mds_part_iii(h_y, part_iii_raw):
            if pd.isna(h_y):
                return pd.NA
            elif h_y < 3:
                return part_iii_raw * 1.2 + 2.3
            elif h_y < 4:
                return part_iii_raw * 1.2 + 1.0
            elif h_y <= 5:
                return part_iii_raw * 1.1 + 7.5
            else:
                return pd.NA

        data['MDS_Part_II'] = data.apply(lambda x: comp_mds_part_ii(h_y=x.h_y, part_ii_raw=x.part_ii_raw), 1)
        data['MDS_Part_III'] = data.apply(lambda x: comp_mds_part_iii(h_y=x.h_y, part_iii_raw=x.part_iii_raw), 1)

        return _fix_column_names(data)

    def import_motor_scores(self):
        mds_scores = self.import_mds_updrs(concise=False)
        old_scores = self.import_old_updrs()

        mds_scores = (mds_scores.filter(items=['session_id', 'UPDRS_date', 'H_Y', 'Part_III']).
                      assign(UPDRS_source='MDS-UPDRS'))

        old_scores = (old_scores.filter(items=['session_id', 'visit_date', 'H_Y', 'MDS_Part_III']).
                      rename(columns={'visit_date': 'UPDRS_date', 'MDS_Part_III': 'Part_III'}).
                      assign(UPDRS_source='UPDRS 1987'))

        data = pd.concat([mds_scores, old_scores], ignore_index=True)

        # some people got multiple UPDRS assessments per overall session
        # (e.g. an abbreviated session triggered a full assessment a few
        # days or weeks later and the UPDRS was done on each occasion). In
        # this case, we delete all but one UPDRS per universal session id,
        # to prevent issues with multiple matches with other data:

        data = (data.groupby(['session_id'], sort=False, group_keys=False).
                apply(lambda x: x.sort_values(['UPDRS_date'], ascending=False)))

        data = (data.groupby(['session_id'], sort=False).
                head(1).reset_index(drop=True).
                drop(columns=['UPDRS_date']))

        return data.fillna(np.nan)

    def import_hads(self, concise=True):
        data = self._load_spreadsheet('hads', na=['NA', 'NA1', 'NA2', 'NA3', 'NA4', 'NA5', 'NA6', '', None])
        data = data.drop(columns=['anxiety_total', 'depression_total'])
        data = self._map_to_universal_session_id(data)

        data = data.assign(HADS_anxiety=data[['q1', 'q3', 'q5', 'q7', 'q9', 'q11', 'q13']].sum(axis=1, skipna=True),
                           HADS_depression=data[['q2', 'q4', 'q6', 'q8', 'q10', 'q12', 'q14']].sum(axis=1, skipna=True))

        data = _fix_column_names(data)
        if concise:
            data = data.filter(items=['session_id', 'HADS_anxiety', 'HADS_depression'])

        return data.fillna(np.nan)

    def import_medications(self, concise=True):
        data = self._load_spreadsheet('meds', na=['NA', 'NA1', 'NA2', 'NA3', 'NA4', 'NA5', 'NA6', '', None])
        data = self._map_to_universal_session_id(data)

        # Convert medication columns to numbers.
        for col in data.columns[data.columns.slice_indexer('selegiline', 'cr_tolcapone')]:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # replace all NAs with 0 in the medication dose columns (from 6 onwards):
        data.loc[:, 'selegiline':'cr_tolcapone'] = data.loc[:, 'selegiline':'cr_tolcapone'].replace([pd.NA, np.nan], 0)

        # calculate the LED subtotal for each type of medication.
        # total immediate release l-dopa - combination of sinemet, madopar, sindopa,
        # kinson (ref = systematic review):
        data = data.assign(ldopa=data.sinemet + data.madopar + data.sindopa + data.kinson,

                           # controlled release l-dopa - combination of sinemet CR and madopar HBS
                           # (ref = systematic review Mov. Disord. 2010. 25(15):2649-2653):
                           cr_ldopa=(data.sinemet_cr + data.madopar_hbs) * 0.75)

        # conversion if COMT inhibitors are taken. Need to convert CR ldopa first
        # then multiply by COMT factor:
        def compute_comp_ir(x):
            if x.ir_entacapone > 0:
                return x.ir_entacapone * 0.33
            elif x.ir_tolcapone > 0:
                return x.ir_tolcapone * 0.5
            else:
                return 0.0

        def comp_comt_cr(x):
            if x.cr_entacapone > 0:
                return x.cr_entacapone * 0.75 * 0.33
            elif x.cr_tolcapone > 0:
                return x.cr_tolcapone * 0.75 * 0.5
            else:
                return 0.0

        data['comt_ir'] = data.apply(lambda x: compute_comp_ir(x), 1)
        data['comt_cr'] = data.apply(lambda x: comp_comt_cr(x), 1)

        # conversion of dopamine agonists and other PD meds
        data = data.assign(amantadine_led=data.amantadine * 1.0,  # ref = syst review
                           apomorphine_led=data.apomorphine * 10.0,  # ref = syst review
                           bromocriptine_led=data.bromocriptine * 10.0,  # ref = syst review (suppl. material)
                           pergolide_led=data.pergolide * 100.0,  # ref - Evans (2004) Mov Disord 19:397-405
                           lisuride_led=data.lisuride * 100.0,  # ref - Parkin (2002) Mov Disord 17:682-692
                           ropinirole_led=data.ropinirole * 20.0,  # ref = syst review
                           pramipexole_led=data.pramipexole * 100.0,  # ref = syst review
                           selegiline_led=data.selegiline * 10.0,  # ref = syst review. This value (* 10) is
                           # for oral formulations. If on sublingual formulations need to use
                           # conversion factor (* 80). Tim does not have any patients taking sublingual
                           # formulations.
                           rotigotine_led=data.rotigotine * 30.0,  # ref = syst review
                           duodopa_led=data.duodopa * 1.11)  # ref = syst review. Only available to those in clinical trial.)

        # Rasagiline (conversion factor * 100) #ref = syst review. Has not been
        # included since it is currently not available in NZ.

        # Anticholinergics are not included in LED calculations but make an indicator
        # column anyway:

        data = data.assign(anticholinergics=(data[['orphenadrine', 'benztropine', 'procyclidine']] > 0).any(axis=1))
        data['anticholinergics'] = data.anticholinergics.astype(
            pd.CategoricalDtype([True, False])).cat.rename_categories({True: 'Yes', False: 'No'})

        data['LED'] = data.apply(lambda x:
                                 x.ldopa + x.cr_ldopa + x.comt_ir + x.comt_cr + x.amantadine_led +
                                 x.apomorphine_led + x.bromocriptine_led + x.pergolide_led +
                                 x.lisuride_led + x.ropinirole_led + x.pramipexole_led +
                                 x.selegiline_led + x.rotigotine_led, 1).round(decimals=2)

        # some people got multiple meds assessments per overall session
        # (e.g. an abbreviated session triggered a full assessment a few
        # days or weeks later and the meds were taken on each occasion). In
        # this case, we delete all but one per universal session id
        # to prevent issues with multiple matches to other data:

        data = (data.groupby(['session_id'], sort=False, group_keys=False).
                apply(lambda x: x.sort_values('med_date', ascending=False)))

        # Select first row.
        data = data.groupby('session_id', sort=False).head(1).reset_index(drop=True)

        if concise:
            data = data.filter(items=['session_id', 'LED'])

        return data.fillna(np.nan)

    def import_neuropsyc(self, concise=True):
        data = self._load_spreadsheet('neuropsyc')

        data = data.rename(columns={'np1_date'           : 'session_date',
                                    'group'              : 'np_group',
                                    'session_type'       : 'full_assessment',
                                    'nzbri_criteria'     : 'cognitive_status',
                                    'global_z_historical': 'global_z_no_language',
                                    'moca'               : 'MoCA',
                                    'wtar_wais_3_fsiq'   : 'WTAR',
                                    'attention_mean'     : 'attention_domain',
                                    'executive_mean'     : 'executive_domain',
                                    'visuo_mean'         : 'visuo_domain',
                                    'memory_mean'        : 'learning_memory_domain',
                                    'language_mean'      : 'language_domain'})

        # Make ordinal data.
        data['cognitive_status'] = (data.cognitive_status.
                                    astype(pd.CategoricalDtype(categories=('U', 'MCI', 'D'), ordered=True)).
                                    cat.rename_categories({'U': 'N'}))

        # Lets assume that all assessments are full, unless explicitly mentioned 'short'.
        data['full_assessment'] = data.apply(lambda x: False if x.full_assessment == 'Short' else True, 1)

        # Compute NP Excluded. Cache is an overkill here, but that's what I'm going with :-D.
        @cache
        def comp_np_exlcuded(neuropsych_excluded):
            # Fixme: replace with switch/case, when python 3.10 is out.
            if neuropsych_excluded == 'Y':
                return True
            elif neuropsych_excluded == 'N':
                return False
            else:
                return pd.NA

        data['np_excluded'] = data.apply(lambda x: comp_np_exlcuded(x.neuropsych_excluded), 1)

        def comp_baseline_values(sub_data):
            # Sort by session date.
            sub_data = sub_data.sort_values('session_date', ascending=True)
            sub_data['session_number'] = np.arange(sub_data.shape[0]) + 1

            baseline_data = (sub_data.head(1).
                             rename(columns={'session_date': 'date_baseline',
                                             'global_z'    : 'global_z_baseline',
                                             'diagnosis'   : 'diagnosis_baseline'}).
                             filter(['subject_id', 'date_baseline', 'global_z_baseline', 'diagnosis_baseline']))

            sub_data = sub_data.merge(baseline_data, how='left')
            return sub_data

        # Create baseline values
        data = data.groupby('subject_id').apply(lambda x: comp_baseline_values(x)).reset_index(drop=True)
        data['years_from_baseline'] = ((data.session_date - data.date_baseline).
                                       astype(np.timedelta64(1, 'D')).
                                       astype('float') / 365.25).round(3)

        FU_latest = data.groupby('subject_id').agg(FU_latest=('years_from_baseline', 'max')).reset_index()

        data = data.merge(FU_latest, how='left').drop(columns=['subject_id', 'session_date'])

        if concise:
            data = data.filter(items=['session_id', 'np_excluded',
                                      'full_assessment', 'diagnosis', 'np_group', 'cognitive_status',
                                      'MoCA', 'WTAR', 'global_z', 'global_z_no_language', 'npi',
                                      'attention_domain', 'executive_domain', 'visuo_domain',
                                      'learning_memory_domain', 'language_domain', 'date_baseline',
                                      'global_z_baseline', 'diagnosis_baseline', 'session_number',
                                      'years_from_baseline', 'FU_latest'])
        else:
            cols = ['session_id', 'np_excluded', 'full_assessment', 'np_group', 'cognitive_status', 'MoCA', 'WTAR']
            cols = cols + [col for col in data.columns.to_list() if col not in cols]
            data = data.filter(items=cols).drop(columns=['session_labels', 'sex', 'age', 'education'])

        return data.fillna(np.nan)

    def import_assessments(self, assessment_type=None):
        data = self._load_spreadsheet(modality='assessments', na=['NA', '', 'None', None])
        data['session_id'] = data.apply(lambda x: f'{x.subject_id}_{x.session_id}', 1)

        if isinstance(assessment_type, str):
            assessment_type = [assessment_type]

        if assessment_type is not None:
            data = data.query(f'assessment_type == {assessment_type}').reset_index(drop=True)

        return data
